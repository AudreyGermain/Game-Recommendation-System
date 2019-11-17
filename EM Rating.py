# create a rating based on time played
game_hrs_density <- function(GAME, nclass, print_vals = TRUE){
    # subsetting data
    game_data <- subset(steam_clean, game1 == GAME & hrs > 2)
game_data$loghrs <- log(game_data$hrs)

# em algorithm
mu.init <- seq(min(game_data$loghrs), max(game_data$loghrs), length = nclass)
EM <- normalmixEM(game_data$loghrs, mu = mu.init, sigma=rep(1, nclass))

# print results
if(print_vals){
cat(" lambda: ", EM$lambda, "\n mean  : ", EM$mu, "\n sigma : ", EM$sigma, "\n")
}

# building data frame for plotting
x <- seq(min(game_data$loghrs), max(game_data$loghrs), 0.01)
dens <- data.frame(x = x)
for(k in 1:nclass){
    dens[,paste0('y', k)] <- EM$lambda[k]*dnorm(x, EM$mu[k], EM$sigma[k])
}
