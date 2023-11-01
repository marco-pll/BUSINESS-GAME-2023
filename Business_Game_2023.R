###########################
# BUSINESS GAME 13 Aprile #
###########################

malware <- read.csv(file.choose())
head(malware)

plot_missing(malware)   


##No variabili costanti
var_costanti <- apply(malware,2,function(x) length(unique(x)))
var_costanti <- names(dati)[var_costanti == 1]
vars <- setdiff(names(dati), var_costanti)
dati <- dati[,vars]

str(malware)

tables <- apply(malware,2,function(x) table(x)[2])

for (i in colnames(malware)){
  malware[,i] <- as.factor(malware[,i])
}

colnames(malware)[1] <- "y"


quant <- NULL
qual <- NULL
for (i in 1:ncol(malware)){
  ifelse(is.numeric(malware[,names(malware)[i]]),quant <- c(quant,names(malware)[i]), qual <- c(qual,names(malware)[i]))
}

quant <- quant[quant!="y"]
qual <- qual[qual!="y"]
esplicative <- c(quant,qual)

##Siamo nel training set, quindi non dobbiamo fare la divisione.



## VALIDATION SET ##
malware.validation <- read.csv(file.choose())
str(malware.validation)

for (i in colnames(malware.validation)){
  malware.validation[,i] <- as.factor(malware.validation[,i])
}

## ANALISI ESPLORATIVA SUL TRAINING SET ##

## Ma che cazzo, è anche già bilanciato.
table(malware$y)
## Non c'è nulla da fare.

corr <- matrix(NA,ncol=length(qual), nrow=1)
colnames(corr) <- qual

for (i in qual){
  v <- cramersV(malware[,i], malware$y)
  corr[,qual==i] <- v
}
sort(corr[,], decreasing =TRUE)
##Le prime sono molto importanti

##Ce ne sono una serie di molto rilevanti, anche singolarmente
table(malware$READ_PHONE_STATE, malware$y)
table(malware$ACCESS_COARSE_LOCATION, malware$y)
table(malware$RECEIVE_BOOT_COMPLETED, malware$y)
table(malware$ACCESS_COARSE_LOCATION, malware$y)
table(malware$ACCESS_COARSE_LOCATION, malware$y)

str(malware)

str(malware.validation)
table(malware.validation$Recent)



## POSSIAMO INIZIARE CON LA MODELLAZIONE ##


cz = function(modello, soglia, y = test$ynum) {
  x = 1 - specificity(y,modello,cutoff = soglia)
  y = sensitivity(y,modello,cutoff = soglia)
  # distanza da (0,1)
  sqrt(((0-x)^2 + (1-y)^2))
}

falsi = function(previsti, osservati){
  n <-  table(previsti,osservati)
  err.tot <- 1 - sum(diag(n))/sum(n)
  fn <- n[1,2]/(n[1,2]+n[2,2])
  fp <- n[2,1]/(n[1,1]+n[2,1])
  return(c("FP" = round(fp,2), "FN" = round(fn,2)))
}





malware$ynum <- as.numeric(malware$y) - 1

f.lin <- formula(paste("ynum ~", paste(esplicative,collapse = " + ")))
f.bin <- formula(paste("y ~", paste(esplicative,collapse = " + ")))

#Le possibili soglie
ss = c(seq(0.01,0.99,by=0.01))

## Separazione in training e test
set.seed(4321)
train <- sample(1:nrow(malware),nrow(malware)*0.6)
test <- setdiff(1:nrow(malware),train)
training <- malware[train,]
test <- malware[test,]

## MODELLO LINEARE, SELEZIONE IN ENTRAMBE LE DIREZIONI PARTENDO DAL MODELLO NULLO ##
m.lin.null <- lm(ynum~1, data=training)
#Formula.lin ? la formula del modello con tutti gli elementi. La selezione ? tramite AIC
m.lin <- step(m.lin.null, scope=f.lin, data = training, direction="both") 
## DEVI METTERE LO STESSO DATASET NEL MODELLO NULLO.


summary(m.lin)

##Previsioni sul dataset di test, per vedere come si comporta.
fits.lin <- predict(m.lin, test)

tab.lin <- table(fits.lin > ss[which.min(sapply(ss, function(l) cz(fits.lin,l)))],test$ynum)

ce.lin <- round(1 - sum(diag(tab.lin))/sum(tab.lin),2)
falsi.lin <- falsi(fits.lin > ss[which.min(sapply(ss, function(l) cz(fits.lin,l)))], test$ynum)


RISULTATI <- data.frame(Nomi = "Modello lineare ridotto", Soglia = ss[which.min(sapply(ss, function(l) cz(fits.lin,l)))], Ce = ce.lin, FP = falsi.lin[1], FN = falsi.lin[2],
                        F1 = f1.lin)
rownames(RISULTATI) <- 1


##Aggiungo anche il recall
##Precision
prec.lin <- tab.lin[2,2]/sum(tab.lin[2,1:2])
recall.lin <- tab.lin[2,2]/sum(tab.lin[1:2,2])
f1.lin <- 2*prec.lin*recall.lin/(prec.lin+recall.lin)




## Passiamo al MODELLO LOGISTICO ##

m.log <- glm(f.bin,family="binomial", data=training)

m.log.null <- glm(y~1,family=binomial, data=training)
m.log.step <- step(m.log.null, scope=f.bin, direction="both") 
summary(m.log.step)

fits.log <- predict(m.log.step,test, type = "response")


tab.log <- table(fits.log > ss[which.min(sapply(ss, function(l) cz(fits.log,l)))],test$ynum)
ce.log <- round(1 - sum(diag(tab.log))/sum(tab.log),2)
falsi.log <- falsi(fits.log > ss[which.min(sapply(ss, function(l) cz(fits.log,l)))], test$ynum)


prec.log <- tab.log[2,2]/sum(tab.log[2,1:2])
recall.log <- tab.log[2,2]/sum(tab.log[1:2,2])
f1.log <- 2*prec.log*recall.log/(prec.log+recall.log)


RISULTATI[2,] <- c("Regressione logistica ridotto", ss[which.min(sapply(ss, function(l) cz(fits.log,l)))],
                   ce.log,falsi.log[1],falsi.log[2], f1.log)





## PASSIAMO AL MODELLO SUCCESSIVO ##

## ANALISI DISCRIMINANTE ##
library(MASS)

#Il modello
m.lda <- lda(f.bin, data=training)
m.lda$means
#La previsione
p.lda <- predict(m.lda,test[,esplicative])$class            #Gi? le classi a cui le osservazioni sono assegnate
fits.lda <- predict(m.lda,test[,esplicative])$posterior[,2]

tab.lda <- table(fits.lda > ss[which.min(sapply(ss, function(l) cz(fits.lda,l)))],test$ynum)
ce.lda <- round(1 - sum(diag(tab.lda))/sum(tab.lda),2)
falsi.lda <- falsi(fits.lda > ss[which.min(sapply(ss, function(l) cz(fits.lda,l)))], test$ynum)


prec.lda <- tab.lda[2,2]/sum(tab.lda[2,1:2])
recall.lda <- tab.lda[2,2]/sum(tab.lda[1:2,2])
f1.lda <- 2*prec.lda*recall.lda/(prec.lda+recall.lda)


RISULTATI[3,] <- c("Analisi discriminante lineare",ss[which.min(sapply(ss, function(l) cz(fits.lda,l)))], 
                   ce.lda,falsi.lda[1],falsi.lda[2], f1.lda)

##Proviamo a ridurre le variabili
selected <- as.character(formula(m.log.step))[3]
selected.1 <- strsplit(as.character(selected)," \\+ ")
var.scelte <- esplicative[esplicative %in% selected.1[[1]]]
f.lda <- formula(paste("y~",paste(var.scelte,collapse=" + ")))


m.lda.r <- lda(f.lda, data=training)
fits.lda.r <- predict(m.lda.r,test[,esplicative])$posterior[,2]

tab.lda.r <- table(fits.lda.r > ss[which.min(sapply(ss, function(l) cz(fits.lda.r,l)))],test$ynum)
ce.lda.r <- round(1 - sum(diag(tab.lda.r))/sum(tab.lda.r),2)
falsi.lda.r <- falsi(fits.lda.r > ss[which.min(sapply(ss, function(l) cz(fits.lda.r,l)))], test$ynum)


prec.lda.r <- tab.lda.r[2,2]/sum(tab.lda.r[2,1:2])
recall.lda.r <- tab.lda.r[2,2]/sum(tab.lda.r[1:2,2])
f1.lda.r <- 2*prec.lda.r*recall.lda.r/(prec.lda.r+recall.lda.r)


RISULTATI[4,] <- c("LDA Ridotta",ss[which.min(sapply(ss, function(l) cz(fits.lda.r,l)))], 
                   ce.lda.r,falsi.lda.r[1],falsi.lda.r[2], f1.lda.r)

lift.roc(fits.lda,test$ynum,type = "crude")


#Per quella quadratica
m.qda <- qda(f.bin, data=training)

p.qda <- predict(m.qda,validation[,esplicative])$class            #Gi? le classi a cui le osservazioni sono assegnate
fits.qda <- predict(m.qda,test[,esplicative])$posterior[,2]

tab.qda <- table(fits.qda > ss[which.min(sapply(ss, function(l) cz(fits.qda,l)))],test$ynum)
ce.qda <- round(1 - sum(diag(tab.qda))/sum(tab.qda),2)
falsi.qda <- falsi(fits.qda > ss[which.min(sapply(ss, function(l) cz(fits.qda,l)))], test$ynum)

prec.qda <- tab.qda[2,2]/sum(tab.qda[2,1:2])
recall.qda <- tab.qda[2,2]/sum(tab.qda[1:2,2])
f1.qda <- 2*prec.qda*recall.qda/(prec.qda+recall.qda)


RISULTATI[5,] <- c("QDA", ss[which.min(sapply(ss, function(l) cz(fits.qda,l)))] ,ce.qda,falsi.qda[1],falsi.qda[2], f1.qda)


## Tutti molto vicini. Proviamo dei modelli diversi ##

### MODELLO GAM ###
### Fare un GAM non ha senso, è il modello lineare se tutte le variabili sono qualitative.


## FACCIAMO UN BEL MARS ##
?write.table

##Questa è la stessa, bene.
log.class <- fits.log > ss[which.min(sapply(ss, function(l) cz(fits.log,l)))]
log.class <- ifelse(log.class==TRUE,1,0)
table(log.class, test$y)
##Previsioni sul validation set.
log.val <- predict(m.log.step,malware.validation, type = "response")
##Utilizzando la stessa soglia decisa sul training set.
log.class.v <- log.val > ss[which.min(sapply(ss, function(l) cz(fits.log,l)))]
log.class.v <- ifelse(log.class.v==TRUE,1,0)



write.table(log.class.v, file=file.choose(),sep = "", quote = FALSE, row.names = FALSE, col.names = FALSE)
dim(malware)
dim(malware.validation)

##Non mi fa inviare ancora i risultati. Peccato
##Procediamo

m.mars <- earth(f.lin, data=training, degree=2, pmethod="cv", nfold = 4)

summary(m.mars)

fits.mars <- predict(m.mars,test[,esplicative])



tab.mars <- table(fits.mars > ss[which.min(sapply(ss, function(l) cz(fits.mars,l)))],test$ynum)
summary(m.mars)
ce.mars <- round(1 - sum(diag(tab.mars))/sum(tab.mars),2)
falsi.mars <- falsi(fits.mars > ss[which.min(sapply(ss, function(l) cz(fits.mars,l)))], test$ynum)


prec.mars <- tab.mars[2,2]/sum(tab.mars[2,1:2])
recall.mars <- tab.mars[2,2]/sum(tab.mars[1:2,2])
f1.mars <- 2*prec.mars*recall.mars/(prec.mars+recall.mars)


##F1 è molto meglio
RISULTATI[6,] <- c("MARS", ss[which.min(sapply(ss, function(l) cz(fits.mars,l)))],ce.mars,falsi.mars[1],falsi.mars[2],
                   f1.mars)


###ALBERO DI CLASSIFICAZIONE##

##Crescita e pruning dell'albero
set.seed(1234)
cb1 <- sample(1:nrow(training), nrow(training)*0.5)
cb2 <- setdiff(1:nrow(training), cb1)

library(tree)
m.tree <- tree(f.bin, data=training[cb1,c("y",esplicative)],
               split = "deviance",
               control=tree.control(nobs=nrow(training[cb1,]),
                                    minsize=5,
                                    mindev=0.001))

#Se volgiamo fare il pruning su un insieme differente.


#Pruning sul secondo insieme. Questo funziona bene nella classificazione.
prune.tree.1= prune.tree(m.tree, newdata=training[cb2,])
plot(prune.tree.1)
abline(v=prune.tree.1$size[which.min(prune.tree.1$dev)], col="red")

#Albero migliore identificato e ottenuto
opt_size <- prune.tree.1$size[which.min(prune.tree.1$dev)]
m.tree.1 <- prune.tree(m.tree, best = opt_size)

fits.tree <- predict(m.tree, test[,esplicative])[,2]

tab.tree <- table(fits.tree > ss[which.min(sapply(ss, function(l) cz(fits.tree,l)))],test$ynum)
summary(m.mars)
ce.tree <- round(1 - sum(diag(tab.tree))/sum(tab.tree),2)
falsi.tree <- falsi(fits.tree > ss[which.min(sapply(ss, function(l) cz(fits.tree,l)))], test$ynum)


prec.tree <- tab.tree[2,2]/sum(tab.tree[2,1:2])
recall.tree <- tab.tree[2,2]/sum(tab.tree[1:2,2])
f1.tree <- 2*prec.tree*recall.tree/(prec.tree+recall.tree)



RISULTATI[7,] <- c("Albero di classificazione", ss[which.min(sapply(ss, function(l) cz(fits.tree,l)))],ce.tree,
                   falsi.tree[1],falsi.tree[2], f1.tree)


lift.roc(fits.tree,test$ynum,type = "crude")

##Proviamo con a convalida incrociata.
m.tree.2 <- tree(f.bin, data=training[,c("y",esplicative)],
               split = "gini",
               control=tree.control(nobs=nrow(training),
                                    minsize=5,
                                    mindev=0.001))
?tree
?cv.tree
prune.tree.2 <- cv.tree(m.tree.2, FUN = prune.tree, K = 10)
plot(prune.tree.2)

##Convalida incrociata fa schifo per la classificazione

m.bag <- bagging(f.bin, data = training, nbagg = 200)
#Per la previsione
fits.bag <- predict(m.bag, test, type ="prob")[,2]
#Da cui possiamo cercare la soglia migliore.

tab.bagg <- table(fits.bag > ss[which.min(sapply(ss, function(l) cz(fits.bag,l)))],test$ynum)
ce.bagg <- round(1 - sum(diag(tab.bagg))/sum(tab.bagg),2)
falsi.bagg <- falsi(fits.bag > ss[which.min(sapply(ss, function(l) cz(fits.bag,l)))], test$ynum)


prec.bagg <- tab.bagg[2,2]/sum(tab.bagg[2,1:2])
recall.bagg <- tab.bagg[2,2]/sum(tab.bagg[1:2,2])
f1.bagg <- 2*prec.bagg*recall.bagg/(prec.bagg+recall.bagg)


##Ottimo improvement
RISULTATI[8,] <- c("Bagging - 200 Alberi", ss[which.min(sapply(ss, function(l) cz(fits.bag,l)))] ,
                   ce.bagg,falsi.bagg[1],falsi.bagg[2], f1.bagg)


##Proviamo a fare di meglio selezionando in modo adattivo il numero di alberi
n <- seq(20,420,by=40)

f1s.bagg <- matrix(NA, nrow=length(n), ncol=1)
rownames(f1s.bagg) <- n
for (i in n) {
  m.bag.tmp <- bagging(f.bin, data = training[cb1,], nbagg = i)
  fits.bag.tmp <- predict(m.bag.tmp, training[cb2,], type ="prob")[,2]
  
  print(i)
  
  tab.bagg.tmp <- table(training$y[cb2],fits.bag.tmp > ss[which.min(sapply(ss, function(l) cz(fits.bag.tmp,l)))])
  prec.bagg.tmp <- tab.bagg.tmp[2,2]/sum(tab.bagg.tmp[2,1:2])
  recall.bagg.tmp <- tab.bagg.tmp[2,2]/sum(tab.bagg.tmp[1:2,2])
  f1.bagg.tmp <- 2*prec.bagg.tmp*recall.bagg.tmp/(prec.bagg.tmp+recall.bagg.tmp)
  
  f1s.bagg[rownames(f1s.bagg) == i,] <- f1.bagg.tmp
}

##Cerchiamo quindi il modello di bagging con il numero di alberi ottimale.
plot(f1s.bagg~n)
n[which.max(f1s.bagg)]





lift.roc(fits.bag,validation$ynum,type = "crude")

##Bagging migliore
m.bag <- bagging(f.bin, data = training, nbagg = 340)
#Per la previsione
fits.bag <- predict(m.bag, test, type ="prob")[,2]
#Da cui possiamo cercare la soglia migliore.

tab.bagg <- table(fits.bag > ss[which.min(sapply(ss, function(l) cz(fits.bag,l)))],test$ynum)
ce.bagg <- round(1 - sum(diag(tab.bagg))/sum(tab.bagg),2)
falsi.bagg <- falsi(fits.bag > ss[which.min(sapply(ss, function(l) cz(fits.bag,l)))], test$ynum)


prec.bagg <- tab.bagg[2,2]/sum(tab.bagg[2,1:2])
recall.bagg <- tab.bagg[2,2]/sum(tab.bagg[1:2,2])
f1.bagg <- 2*prec.bagg*recall.bagg/(prec.bagg+recall.bagg)


##Ottimo improvement
RISULTATI[8,] <- c("Bagging - 340 Alberi", ss[which.min(sapply(ss, function(l) cz(fits.bag,l)))] ,
                   ce.bagg,falsi.bagg[1],falsi.bagg[2], f1.bagg)







## Iniziamo a scrivere le cose per il modello RANDOM FOREST ##

oob = NULL
F = c(3,4,5,6,7,8,9, 10, 11,12,17)
for(f in F){  rfModel = randomForest(y = training$y, x = training[,esplicative], mtry = f, ntree=300, do.trace=50)
oob = rbind(oob, c("f"=f, "err.OOB"=rfModel$err.rate[nrow(rfModel$err.rate),1]))
}
#Ci salviamo l'errore out-of-bootstrap
#Lo usiamo per trovare un modello ottimale
plot(oob, xlab="variabili campionate", ylab="Errore OOB", type="b")
abline(v=F[which.min(oob[,2])], col=4)


m.rf <- randomForest(f.bin, data = training, mtry = F[which.min(oob[,2])], ntree=800)
fits.rf <- predict(m.rf,test, type="prob")[,2]

tab.rf <- table(fits.rf > ss[which.min(sapply(ss, function(l) cz(fits.rf,l)))],test$ynum)
summary(m.mars)
ce.rf <- round(1 - sum(diag(tab.rf))/sum(tab.rf),2)
falsi.rf <- falsi(fits.rf > ss[which.min(sapply(ss, function(l) cz(fits.rf,l)))], test$ynum)


prec.rf <- tab.rf[2,2]/sum(tab.rf[2,1:2])
recall.rf <- tab.rf[2,2]/sum(tab.rf[1:2,2])
f1.rf <- 2*prec.rf*recall.rf/(prec.rf+recall.rf)



RISULTATI[9,] <- c("Random forest - 10 vars.", ss[which.min(sapply(ss, function(l) cz(fits.rf,l)))] ,
                   ce.rf,falsi.rf[1],falsi.rf[2], f1.rf)

lift.roc(fits.rf,test$ynum,type = "crude")


fits.rf.validation <- predict(m.rf,malware.validation, type="prob")[,2]

rf.class.v <- fits.rf.validation > ss[which.min(sapply(ss, function(l) cz(fits.rf,l)))]
rf.class.v <- ifelse(rf.class.v==TRUE,1,0)
write.table(rf.class.v, file=file.choose(),sep = "", quote = FALSE, row.names = FALSE, col.names = FALSE)



##Proviamo a fare convalida incrociata per la random forest?
library(randomForest)
oob = NULL
F = c(10, 11,12,13,14,15,16,17,18,19,20)
nsets <- 10
f1s <- matrix(NA, nrow=length(F), ncol=nsets)

set.seed(4)
cv_rf <- function () {
  
  order <- sample(1:nrow(training), nrow(training))
  dim_subset <- floor(nrow(training)/nsets)
  for (i in 1:nsets) {
    test_set <- ((i-1)*dim_subset + 1):(i*dim_subset)
    test_set <- order[test_set]
    train_set <- order[-test_set]
    
    for (f in F) {
      rfModel = randomForest(y = training$y[train_set], x = training[train_set,esplicative], mtry = f, ntree=200, do.trace=50)
      fits.rf <- predict(rfModel, training[test_set,esplicative], type ="prob")[,2]
      p1n <- fits.rf > ss[which.min(sapply(ss, function(l) cz(fits.rf,l)))]
      
      true.pos <- sum(p1n==1 & training$y[test_set]==1)
      false.pos <- sum(p1n==1 & training$y[test_set]==0)
      false.neg <- sum(p1n==0 &  training$y[test_set]==1)
      prec.n1 <- true.pos/(true.pos + false.pos)
      recall.n1 <- true.pos/(true.pos + false.neg)
      f1.n1 <- 2*prec.n1*recall.n1/(prec.n1+recall.n1)
      
      f1s[F==f, i] <- f1.n1
    }
  }
  
  return(f1s)
}

f1s_returned <- cv_rf()

f1s_mean <- apply(f1s_returned,1,mean)
plot(f1s_mean~F)


m.rf <- randomForest(f.bin, data = training, mtry = 11, ntree=600, do.trace = 50)
fits.rf <- predict(m.rf,test, type="prob")[,2]

tab.rf <- table(fits.rf > ss[which.min(sapply(ss, function(l) cz(fits.rf,l)))],test$ynum)
summary(m.mars)
ce.rf <- round(1 - sum(diag(tab.rf))/sum(tab.rf),2)
falsi.rf <- falsi(fits.rf > ss[which.min(sapply(ss, function(l) cz(fits.rf,l)))], test$ynum)


prec.rf <- tab.rf[2,2]/sum(tab.rf[2,1:2])
recall.rf <- tab.rf[2,2]/sum(tab.rf[1:2,2])
f1.rf <- 2*prec.rf*recall.rf/(prec.rf+recall.rf)



RISULTATI[9,] <- c("Random forest - 14 vars. - 350 Alberi", ss[which.min(sapply(ss, function(l) cz(fits.rf,l)))] ,
                   ce.rf,falsi.rf[1],falsi.rf[2], f1.rf)



fits.rf.validation <- predict(m.rf,malware.validation, type="prob")[,2]

rf.class.v <- fits.rf.validation > ss[which.min(sapply(ss, function(l) cz(fits.rf,l)))]
rf.class.v <- ifelse(rf.class.v==TRUE,1,0)
write.table(rf.class.v, file=file.choose(),sep = "", quote = FALSE, row.names = FALSE, col.names = FALSE)







  ####   BOOSTING   ###



##Proviamo con i prossimi modelli.
library(ada)
#Con stumps, su campione bilanciato
#Qui la risposta ? quantitativa.
m_boost = ada(f.lin, data = training,  iter = 300,
              rpart.control(maxdepth=1,cp=-1,minsplit=0,xval=0))
# Gli argomenti test.x e test.y servono per valutare quando arrestare l'algoritmo.
# Se non specificati, la suddivisione ? automatica.

?ada
m_boost = ada(f.lin, data = training, loss = "l2", iter = 300,
               rpart.control(maxdepth=2,cp=-1,minsplit=0,xval=0))

#Misure di importanza credo delle variabili.
varplot(m_boost)
varplot(m_boostI)

#Per le previsioni
fits.boost = predict(m_boost, test,type = "prob")[,2]

tab.boost <- table(fits.boost > ss[which.min(sapply(ss, function(l) cz(fits.boost,l)))],test$ynum)
summary(m.mars)
ce.boost <- round(1 - sum(diag(tab.boost))/sum(tab.boost),2)
falsi.boost <- falsi(fits.boost > ss[which.min(sapply(ss, function(l) cz(fits.boost,l)))], test$ynum)


prec.boost <- tab.boost[2,2]/sum(tab.boost[2,1:2])
recall.boost <- tab.boost[2,2]/sum(tab.boost[1:2,2])
f1.boost <- 2*prec.boost*recall.boost/(prec.boost+recall.boost)



RISULTATI[10,] <- c("Boosting-  300 alberi", ss[which.min(sapply(ss, function(l) cz(fits.boost,l)))] ,
                    ce.boost,falsi.boost[1],falsi.boost[2], f1.boost)

lift.roc(fits.boost,validation$ynum,type = "crude")

fits.boost.validation <- predict(m_boost,malware.validation, type="prob")[,2]

boost.class.v <- fits.boost.validation > ss[which.min(sapply(ss, function(l) cz(fits.boost,l)))]
boost.class.v <- ifelse(boost.class.v==TRUE,1,0)
write.table(boost.class.v, file=file.choose(),sep = "", quote = FALSE, row.names = FALSE, col.names = FALSE)




##Il boosting è simile al modello gradient boosting? Se sì, allora possiamo cercare un numero dideale di alberi.

n <- seq(150,300,by=30)

f1s.boost <- matrix(NA, nrow=length(n), ncol=1)
rownames(f1s.boost) <- n
for (i in n) {
  m.boost.tmp <- ada(f.lin, data = training[cb1,],  iter = i,
                   rpart.control(maxdepth=1,cp=-1,minsplit=0,xval=0))
  fits.boost.tmp <- predict(m.boost.tmp, training[cb2,], type ="prob")[,2]
  
  print(i)
  
  tab.boost.tmp <- table(training$y[cb2],fits.boost.tmp > ss[which.min(sapply(ss, function(l) cz(fits.boost.tmp,l)))])
  prec.boost.tmp <- tab.boost.tmp[2,2]/sum(tab.boost.tmp[2,1:2])
  recall.boost.tmp <- tab.boost.tmp[2,2]/sum(tab.boost.tmp[1:2,2])
  f1.boost.tmp <- 2*prec.boost.tmp*recall.boost.tmp/(prec.boost.tmp+recall.boost.tmp)
  
  f1s.boost[rownames(f1s.boost) == i,] <- f1.boost.tmp
}

plot(f1s.boost~n)



## Nel mentre, prepariamo la SVM ##

library(e1071)
m.svm = svm(f.bin, data = training,  kernel="sigmoid")

#Questa funzione fa un tuning tramite separazione in training e test set.
?tune
svm_t = tune(svm, f.bin,data = training,  kernel="sigmoid",
             ranges=list(cost=2^(-5:1)), tunecontrol=tune.control(cross=5))

#Separazione in training e test automatica. Funziona bene comunque
svm_t = tune(svm, f.bin,data = training,  kernel="sigmoid", ranges=list(cost=2^(-5:1)))


#Tune fa una separazione in training e test set per trovare il modello ottimale.
plot(svm_t)
abline(v=1  , col="red")
svm_t$best.model              #Per vedere qual'? il costo migliore

#Con il modello regolato.
m_svm = svm(f.bin,data = training,  kernel="sigmoid",
            cost = 1,
            probability = T)


fits.svm = attr(predict(m_svm,test,probability = T),"probabilities")
fits.svm <- fits.svm[,1]

tab.svm <- table(fits.svm > ss[which.min(sapply(ss, function(l) cz(fits.svm,l)))], test$ynum)
ce.svm <- round(1 - sum(diag(tab.svm))/sum(tab.svm),2)
falsi.svm <- falsi(fits.svm > ss[which.min(sapply(ss, function(l) cz(fits.svm,l)))], test$ynum)


prec.svm <- tab.svm[2,2]/sum(tab.svm[2,1:2])
recall.svm <- tab.svm[2,2]/sum(tab.svm[1:2,2])
f1.svm <- 2*prec.svm*recall.svm/(prec.svm+recall.svm)



RISULTATI[13,] <- c("Support vector machine", ss[which.min(sapply(ss, function(l) cz(fits.svm,l)))] ,
                    ce.svm,falsi.svm[1],falsi.svm[2], f1.svm)

     #### RETE NEURALE ###

library(nnet)

#Lista dei parametri di regolazione (decay)
decay <- 10^(seq(-4, -2, length=10))
nodi <- c(2,3,4,5,6)

#Cerchaimo il valore ottimale di nodi e decay. Usiamo l'f1 score.
err <- matrix(NA,50,4)
z <- matrix(NA,nrow=5,ncol=10)

set.seed(1234)
i=1
{
  print(c("i", "errore", "decay", "nodi"))
  for(k in 1:10){
    for(j in 1:5){
      #Risposta direttamente qualitativa.
      n1<- nnet(y~., data=training[cb1,c("y",esplicative)], decay=decay[k], size=nodi[j],
                maxit=200,  trace=TRUE, MaxNWts=500)
      p1n<- predict(n1, newdata=training[cb2,], type="class") # "class"
      
      a<- table(p1n, training[cb2,"y"])
      true.pos <- sum(p1n==1 & training$y[cb2]==1)
      false.pos <- sum(p1n==1 & training$y[cb2]==0)
      false.neg <- sum(p1n==0 &  training$y[cb2]==1)
      prec.n1 <- true.pos/(true.pos + false.pos)
      recall.n1 <- true.pos/(true.pos + false.neg)
      f1.n1 <- 2*prec.n1*recall.n1/(prec.n1+recall.n1)
      err[i,]<-c(i,f1.n1,decay[k], nodi[j])
      z[j,k]<- f1.n1
      print(err[i,])
      i<-i+1
    }
  }
}

#E poi per trovare il migliore
massimo<-which.max(err[,2])
err[massimo,c(3,4)]
plot(err[,2], main = "Selezione per la rete neurale", ylab="F1 score", xlab="indice")
abline(v=massimo,col=2)

#E quindi possiamo stimare il modello ottimale
nn1 <- nnet(y~., data = training[,c("y",esplicative)], size=err[massimo,4], decay = err[massimo,3], maxit=1500,  MaxNWts=500)
pp1 <- predict(nn1, newdata=test, type="class")

#E poi calcoliamo gli indici.
fits.nn = predict(nn1, newdata=test, type="raw")[,1]
head(fits.nn)
tab.nn <- table(fits.nn > ss[which.min(sapply(ss, function(l) cz(fits.nn,l)))], test$ynum)
ce.nn <- round(1 - sum(diag(tab.nn))/sum(tab.nn),2)
falsi.nn <- falsi(fits.nn > ss[which.min(sapply(ss, function(l) cz(fits.nn,l)))], test$ynum)



prec.nn <- tab.nn[2,2]/sum(tab.nn[2,1:2])
recall.nn <- tab.nn[2,2]/sum(tab.nn[1:2,2])
f1.nn <- 2*prec.nn*recall.nn/(prec.nn+recall.nn)


RISULTATI[11,] <- c("Rete neurale", ss[which.min(sapply(ss, function(l) cz(fits.nn,l)))] ,
                    ce.nn,falsi.nn[1],falsi.nn[2], f1.nn)


### RISULTATI FINALI::
### SUL MIO TEST SET, ANCHE CAMBIANDOLO, L'F1 SCORE RIMANE 0.907 ##
### Quindi buonissimo. Ho fatto anche 0.911 con lo stesso modello.


















































