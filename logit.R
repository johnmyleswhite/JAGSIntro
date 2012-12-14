library("rjags")

df <- read.csv("logit.csv")

jags <- jags.model("logit.bugs",
                   data = list("x" = with(df, X),
                               "y" = with(df, Y),
                               "N" = nrow(df)),
                   n.chains = 4,
                   n.adapt = 0)
 
mcmc.samples <- coda.samples(jags,
                        c("a", "b"),
                        50)

png("pre_burnin.png")
plot(mcmc.samples)
dev.off()

jags <- jags.model("logit.bugs",
                   data = list("x" = with(df, X),
                               "y" = with(df, Y),
                               "N" = nrow(df)),
                   n.chains = 4,
                   n.adapt = 1000)
 
mcmc.samples <- coda.samples(jags,
                        c("a", "b"),
                        5000)

png("post_burnin.png")
plot(mcmc.samples)
dev.off()

summary(mcmc.samples)
