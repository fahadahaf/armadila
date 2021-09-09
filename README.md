# armadila
**In Progress**  

Here's an implementation of ARIMADL model that combines Distributed Lags (DLs) with the regular Auto Regressive Integrated Moving Average (ARMA/ARIMA) model. This is essentially a wrapper for the *auto_arima* functionality of **pmdarima** python library where model selection can be performed on ARMA with distributed lags. 

The expression for ARMA (AR + MA) model is given below:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\bg_white&space;X_{t}=c&plus;\varepsilon&space;_{t}&plus;\sum&space;_{i=1}^{p}\varphi&space;_{i}X_{t-i}&plus;\sum&space;_{i=1}^{q}\theta&space;_{i}\varepsilon&space;_{t-i}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;X_{t}=c&plus;\varepsilon&space;_{t}&plus;\sum&space;_{i=1}^{p}\varphi&space;_{i}X_{t-i}&plus;\sum&space;_{i=1}^{q}\theta&space;_{i}\varepsilon&space;_{t-i}" title="X_{t}=c+\varepsilon _{t}+\sum _{i=1}^{p}\varphi _{i}X_{t-i}+\sum _{i=1}^{q}\theta _{i}\varepsilon _{t-i}" /></a>  

For <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;k" title="k" /> distributed lags, ARDL model can be expressed as:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\bg_white&space;X_{t}=\varphi_{0}&plus;\varepsilon&space;_{t}&plus;\sum&space;_{i=1}^{p}\varphi&space;_{i}X_{t-i}&plus;\sum_{i=0}^{k}\alpha_{i}x_{t-i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\bg_white&space;X_{t}=\varphi_{0}&plus;\varepsilon&space;_{t}&plus;\sum&space;_{i=1}^{p}\varphi&space;_{i}X_{t-i}&plus;\sum_{i=0}^{k}\alpha_{i}x_{t-i}" title="\bg_white X_{t}=\varphi_{0}+\varepsilon _{t}+\sum _{i=1}^{p}\varphi _{i}X_{t-i}+\sum_{i=0}^{k}\alpha_{i}x_{t-i}" /></a>

Combining distributed lags (DLs) with the ARMA/ARIMA model, we get:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\bg_white&space;X_{t}=c&plus;\varepsilon&space;_{t}&plus;\sum&space;_{i=1}^{p}\varphi&space;_{i}X_{t-i}&plus;\sum&space;_{i=1}^{q}\theta&space;_{i}\varepsilon&space;_{t-i}&plus;\sum_{j=0}^{n-1}\sum_{i=0}^{k}\alpha_{i}^{j}x_{t-i}^{j}," target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;X_{t}=c&plus;\varepsilon&space;_{t}&plus;\sum&space;_{i=1}^{p}\varphi&space;_{i}X_{t-i}&plus;\sum&space;_{i=1}^{q}\theta&space;_{i}\varepsilon&space;_{t-i}&plus;\sum_{j=0}^{n-1}\sum_{i=0}^{k}\alpha_{i}^{j}x_{t-i}^{j}," title="X_{t}=c+\varepsilon _{t}+\sum _{i=1}^{p}\varphi _{i}X_{t-i}+\sum _{i=1}^{q}\theta _{i}\varepsilon _{t-i}+\sum_{j=0}^{n-1}\sum_{i=0}^{k}\alpha_{i}^{j}x_{t-i}^{j}," /></a>  
where <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;n" title="n" /> is the number of exogenous variables.
