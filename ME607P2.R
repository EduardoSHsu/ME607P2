# PROJETO 2 (P2) - ANÁLISE DE DADOS DE SÉRIES TEMPORAIS

# Lista de pacotes necessários 
pacotes <- c("tidyverse", "ggplot2", "lubridate", "moments", "FinTS", "forecast", "tseries", "patchwork", "lmtest")

for (p in pacotes) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

# Carregamento dos Dados
url_dados <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
df <- read.csv(url_dados)

# Limpeza e Transformação
df_clean <- df %>%
  mutate(DataHora = make_datetime(year, month, day, hour)) %>%
  filter(!is.na(pm2.5)) %>%
  select(DataHora, year, month, pm2.5)

# Agregação para média diária
df_daily <- df_clean %>%
  mutate(Data = as.Date(DataHora)) %>%
  group_by(Data, year, month) %>%
  summarise(PM25 = mean(pm2.5), .groups = 'drop')

# Estatísticas Descritivas Básicas
media <- mean(df_daily$PM25)
desvio <- sd(df_daily$PM25)
assimetria <- skewness(df_daily$PM25)
curtose <- kurtosis(df_daily$PM25)    

cat("\n--- Estatísticas Descritivas ---\n")
cat("Média:", media, "\n")
cat("Desvio Padrão:", desvio, "\n")
cat("Assimetria (Skewness):", assimetria, "\n")
cat("Curtose:", curtose, "\n")

# Gráfico da Série no Tempo 
g1 <- ggplot(df_daily, aes(x = Data, y = PM25)) +
  geom_line(color = "steelblue", alpha = 0.8) +
  theme_minimal() +
  labs(title = "",
       x = "Ano", y = "Concentração de PM2.5")

# Histograma + Densidade 
g2 <- ggplot(df_daily, aes(x = PM25)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, fill = "lightgray", color = "black") +
  geom_density(color = "red", linewidth = 1) +
  theme_minimal() +
  labs(title = "",
       x = "PM2.5", y = "Densidade")

# Sazonalidade via Boxplot 
df_daily$month_factor <- as.factor(df_daily$month)
g3 <- ggplot(df_daily, aes(x = month_factor, y = PM25, fill = month_factor)) +
  geom_boxplot(alpha = 0.7, show.legend = FALSE) +
  theme_minimal() +
  scale_fill_viridis_d() +
  labs(title = "",
       x = "Mês", y = "PM2.5")

# Decomposição Clássica da Série Temporal 
ts_exploratoria <- ts(df_daily$PM25, frequency = 365) 
decomposicao <- stl(ts_exploratoria, s.window = "periodic")

g4 <- autoplot(decomposicao) +
  theme_minimal() +
  labs(title = "",
       subtitle = "")

# Exibindo os gráficos exploratórios
print(g1)
print(g2)
print(g3)
print(g4)

# ESTABILIZAÇÃO DA VARIÂNCIA 
# Transformação
df_daily <- df_daily %>%
  mutate(PM25_log = log(PM25 + 1))

# Teste de Estacionariedade na Série Logarítmica
cat("\n--- Teste ADF: Série Transformada (Apenas Log) ---\n")
print(adf.test(df_daily$PM25_log, alternative = "stationary"))

# Calculando a variância por mês para ver a estabilidade
variancia_mensal <- df_daily %>%
  group_by(year, month) %>%
  summarise(
    Data = first(Data),
    Var_Original = var(PM25, na.rm = TRUE),
    Var_Transformada = var(PM25_log, na.rm = TRUE),
    .groups = 'drop'
  )

# Gráfico da Variância Mensal
g_var_original <- ggplot(variancia_mensal, aes(x = Data, y = Var_Original)) +
  geom_line(color = "darkred", linewidth = 1) +
  theme_minimal() +
  labs(title = "", subtitle = "", x = "Tempo", y = "Variância do PM2.5")

g_var_transformada <- ggplot(variancia_mensal, aes(x = Data, y = Var_Transformada)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  theme_minimal() +
  labs(title = "", subtitle = "", x = "Tempo", y = "Variância do log(PM2.5)")

print(g_var_original / g_var_transformada)

# IDENTIFICAÇÃO E SELEÇÃO DE MODELOS 
# Gráficos de Autocorrelação (ACF e PACF) para Identificação
ts_estacionaria <- ts(df_daily$PM25_log, frequency = 365)

g_fac <- ggAcf(ts_estacionaria, lag.max = 30) + 
  theme_minimal() + 
  labs(title = "", subtitle = "")

g_facp <- ggPacf(ts_estacionaria, lag.max = 30) + 
  theme_minimal() + 
  labs(title = "", subtitle = "")

print(g_fac / g_facp)

# Estimação de modelos concorrentes 
modelo_ar2    <- Arima(df_daily$PM25_log, order = c(2, 0, 0))
modelo_ma3    <- Arima(df_daily$PM25_log, order = c(0, 0, 3))
modelo_arma11 <- Arima(df_daily$PM25_log, order = c(1, 0, 1))
modelo_arma22 <- Arima(df_daily$PM25_log, order = c(2, 0, 2))

# Estimação do modelo SARIMA anual
onda_sazonal <- fourier(ts_estacionaria, K = 1)
modelo_sarima_anual <- Arima(ts_estacionaria, order = c(1, 0, 1), xreg = onda_sazonal)

# Avaliação de Desempenho (AIC)
cat("\n--- Comparação de AIC (Modelos Estacionários) ---\n")
cat("AIC do AR(2):      ", AIC(modelo_ar2), "\n")
cat("AIC do MA(3):      ", AIC(modelo_ma3), "\n")
cat("AIC do ARMA(1,1):  ", AIC(modelo_arma11), "\n")
cat("AIC do ARMA(2,2):  ", AIC(modelo_arma22), "\n")
cat("AIC do ARMA(1,1) Sazonal: ", AIC(modelo_sarima_anual), "\n")

# Escolha do modelo vencedor 
modelo_final <- modelo_arma11

cat("\n--- Teste de Significância dos Parâmetros: Modelo Final ---\n")
print(coeftest(modelo_final))


# ANÁLISE DE RESÍDUOS
residuos_finais <- residuals(modelo_final)

# Gráfico da Série de Resíduos no Tempo
g_res_tempo <- ggplot(data.frame(Data = df_daily$Data, Res = as.numeric(residuos_finais)), aes(x = Data, y = Res)) +
  geom_line(color = "gray40", linewidth = 0.5) +
  theme_minimal() +
  labs(title = NULL, x = "Tempo", y = "Resíduos") 

# ACF dos Resíduos 
g_res_acf <- ggAcf(residuos_finais, lag.max = 30) +
  theme_minimal() +
  labs(title = NULL, subtitle = NULL, y = "ACF", x = "Lag")

# Histograma dos Resíduos com Curva Normal
g_res_hist <- ggplot(data.frame(Res = as.numeric(residuos_finais)), aes(x = Res)) +
  geom_histogram(aes(y = after_stat(density)), bins = 40, fill = "lightgray", color = "black") +
  stat_function(fun = dnorm, args = list(mean = mean(residuos_finais, na.rm = TRUE), sd = sd(residuos_finais, na.rm = TRUE)), color = "blue", linewidth = 1) +
  theme_minimal() +
  labs(title = NULL, x = "Resíduos", y = "Densidade")

painel_checkresiduals <- g_res_tempo / (g_res_acf | g_res_hist)
print(painel_checkresiduals)

# Teste Estatístico da Média 
teste_media <- Box.test(residuos_finais, lag = 10, type = "Ljung-Box")
cat("\n--- Teste Ljung-Box (Média) ---\n")
print(teste_media)

# ACF dos Resíduos ao QUADRADO 
g_res_quadrados_fac <- ggAcf(residuos_finais^2, lag.max = 30) +
  theme_minimal() +
  geom_hline(yintercept = c(-1.96/sqrt(length(residuos_finais)), 1.96/sqrt(length(residuos_finais))), 
             color = "red", linetype = "dashed") +
  labs(title = NULL, subtitle = NULL, y = "ACF (Resíduos Quadrados)", x = "Lag")
print(g_res_quadrados_fac)

# Teste ARCH-LM
teste_arch <- ArchTest(residuos_finais, lags = 10)
cat("\n--- Teste ARCH-LM de Engle (Heterocedasticidade Condicional) ---\n")
print(teste_arch)

# Teste de normalidade
teste_jb_residuos <- jarque.bera.test(na.omit(as.numeric(residuos_finais)))
cat("\n--- Teste Jarque Bera (Normalidade) ---\n")
print(teste_jb_residuos)


# VALIDAÇÃO CRUZADA E BACKTESTING
# CENÁRIO DE VERÃO
data_corte_verao <- as.Date("2014-07-01")

treino_verao <- df_daily %>% filter(Data < data_corte_verao)
reais_verao  <- df_daily %>% filter(Data >= data_corte_verao & Data < data_corte_verao + 15)

# Reestimação do modelo selecionado
modelo_t_verao <- Arima(treino_verao$PM25_log, order = c(1, 0, 1))
prev_verao_log <- forecast(modelo_t_verao, h = 15)

df_prev_verao <- data.frame(
  Data     = reais_verao$Data,
  Real     = reais_verao$PM25,
  Previsto = as.numeric(exp(prev_verao_log$mean) - 1),
  Inf_95   = as.numeric(exp(prev_verao_log$lower[, "95%"]) - 1),
  Sup_95   = as.numeric(exp(prev_verao_log$upper[, "95%"]) - 1)
)

g_prev_verao <- ggplot(df_prev_verao, aes(x = Data)) +
  geom_ribbon(aes(ymin = Inf_95, ymax = Sup_95), fill = "blue", alpha = 0.1) +
  geom_line(aes(y = Real, color = "Real"), linewidth = 1) +
  geom_line(aes(y = Previsto, color = "Previsto"), linewidth = 1, linetype = "dashed") +
  theme_minimal() +
  scale_color_manual(values = c("Real" = "black", "Previsto" = "blue")) +
  labs(title = "Previsão de 15 Dias: Verão (Julho/2014)", x = "Data", y = "PM2.5", color = "Série")


# CENÁRIO DE INVERNO
data_corte_inverno <- as.Date("2014-01-01")

treino_inverno <- df_daily %>% filter(Data < data_corte_inverno)
reais_inverno  <- df_daily %>% filter(Data >= data_corte_inverno & Data < data_corte_inverno + 15)

# Reestimação do modelo selecionado
modelo_t_inverno <- Arima(treino_inverno$PM25_log, order = c(1, 0, 1))
prev_inverno_log <- forecast(modelo_t_inverno, h = 15)

df_prev_inverno <- data.frame(
  Data     = reais_inverno$Data,
  Real     = reais_inverno$PM25,
  Previsto = as.numeric(exp(prev_inverno_log$mean) - 1),
  Inf_95   = as.numeric(exp(prev_inverno_log$lower[, "95%"]) - 1),
  Sup_95   = as.numeric(exp(prev_inverno_log$upper[, "95%"]) - 1)
)

g_prev_inverno <- ggplot(df_prev_inverno, aes(x = Data)) +
  geom_ribbon(aes(ymin = Inf_95, ymax = Sup_95), fill = "red", alpha = 0.1) +
  geom_line(aes(y = Real, color = "Real"), linewidth = 1) +
  geom_line(aes(y = Previsto, color = "Previsto"), linewidth = 1, linetype = "dashed") +
  theme_minimal() +
  scale_color_manual(values = c("Real" = "black", "Previsto" = "red")) +
  labs(title = "Previsão de 15 Dias: Inverno (Janeiro/2014)", x = "Data", y = "PM2.5", color = "Série")

print(g_prev_verao / g_prev_inverno)

# AVALIAÇÃO DE DESEMPENHO PREDITIVO 
# Métricas para o Cenário de Verão
rmse_verao <- sqrt(mean((df_prev_verao$Real - df_prev_verao$Previsto)^2))
mae_verao <- mean(abs(df_prev_verao$Real - df_prev_verao$Previsto))
cobertura_verao <- mean(df_prev_verao$Real >= df_prev_verao$Inf_95 & df_prev_verao$Real <= df_prev_verao$Sup_95) * 100

# Métricas para o Cenário de Inverno
rmse_inverno <- sqrt(mean((df_prev_inverno$Real - df_prev_inverno$Previsto)^2))
mae_inverno <- mean(abs(df_prev_inverno$Real - df_prev_inverno$Previsto))
cobertura_inverno <- mean(df_prev_inverno$Real >= df_prev_inverno$Inf_95 & df_prev_inverno$Real <= df_prev_inverno$Sup_95) * 100


cat("\n--- Métricas de Erro e Cobertura (Backtesting de 15 Dias) ---\n")
cat("VERÃO (Julho):\n")
cat("RMSE:      ", round(rmse_verao, 2), "\n")
cat("MAE:       ", round(mae_verao, 2), "\n")
cat("Cobertura: ", round(cobertura_verao, 2), "%\n\n")

cat("INVERNO (Janeiro):\n")
cat("RMSE:      ", round(rmse_inverno, 2), "\n")
cat("MAE:       ", round(mae_inverno, 2), "\n")
cat("Cobertura: ", round(cobertura_inverno, 2), "%\n")