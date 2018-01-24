# Sky Cast
## Machine Learning Nanodegree: Capstone Project
#### Author: Franklin Bradfield

Forecasting the future remains one of the most challenging problems in machine learning and data science in general. Recent successes toward this endeavor have utilized sequence-to-sequence neural network models, which is where I took inspiration from to produce this project. [Here](https://github.com/Arturus/kaggle-web-traffic) is a recent example the winning solution to a Kaggle competition that involved such an approach to forecast web time series.

In this machine learning project, I compare the performance of a traditional statistical technique, ARIMA, and recurrent neural networks (RNNs) on forecasting commercial airline data that I scraped from the U.S. Department of Transportation's [publicly available sources](https://www.transtats.bts.gov/Data_Elements.aspx).

The code for this was written in Python 2.7, using the latest versions of Python's standard scientific computing libraries - Numpy, Pandas, Matplotlib, and Scikit-Learn, as well as statsmodels - Python's statistical modeling library, and Tensorflow - Google's deep learning library.

To reproduce the results, or to simply view and manipulate the code on your own machine, first setup and activate an environment with the required libraries below installed, then *$ cd* into the notebooks directory and run *$ jupyter notebook*. Exploratory data analysis can be found in data\_analysis.ipynb, while ARIMA\_time\_series.ipynb and RNN\_time\_series.ipynb contain time series models these models respectively.

#### Requirements

- [Python 2.7](https://www.python.org/download/releases/2.7/) --> 2.7.14
- [numpy](http://www.numpy.org/) --> 1.14.0
- [matplotlib](https://matplotlib.org/) --> 2.0.2
- [pandas](http://pandas.pydata.org/) --> 0.22.0
- [scikit-learn](http://scikit-learn.org/stable/) --> 0.19.1
- [statsmodels](http://www.statsmodels.org/dev/index.html) --> 0.8.0
- [Tensorflow](https://www.tensorflow.org/) --> 1.4.1

### Further Reading

**ARIMA for Time Series:**

- Aas, K., & Dimakos, X. K. (2004). Statistical modelling of financial
time series. Norwegian Computing Center. Retrived from
https://www.nr.no/files/samba/bff/SAMBA0804.pdf

- Adebiyi, A. A., Adewumi, A. O., & Ayo, C. K. (2014). Stock Price Prediction
Using the ARIMA Model. International Conference on Computer
Modelling and Simulation 2014. Retrived from 
http://ijssst.info/Vol-15/No-4/data/4923a105.pdf

- Dickey, D. A.; Fuller, W. A. (1979). Distribution of the estimators for
autoregressive time series with a unit root. Journal of the American Statistical
Association. 74(366), 427–431. JSTOR 2286348. doi:10.2307/2286348.

- Kohzadi, N., Boyd, M. S., Kermanshahi, B., & Kaastra, I. (1996). A
comparison of artificial neural network and time series models for forecasting
commodity prices. Neurocomputing, 10(2), 169-181. Retrieved from
http://www.sciencedirect.com/science/article/pii/0925231295000208

**Recurrent Neural Networks for Time Series:**

- Falode, O., & Udomboso, C. (2016). Predictive Modeling of Gas Production,
Utilization and Flaring in Nigeria using TSRM and TSNN: A Comparative
Approach. Open Journal of Statistics, 6(1), 194-207. Retrieved from
http://www.scirp.org/journal/PaperInformation.aspx?PaperID=63994

- Gers, F. A., Eck, D., & Schmidhuber, J. (2001). Applying LSTM to
time series predictable through time-window approaches. International
Conference on Artificial Networks 2001 (pp. 669-676). Retrieved from
https://link.springer.com/chapter/10.1007/3-540-44668-0_93.

- Lipton, Z. C., Kale, D. C., Elkan, C., & Wetzel, R. (2016). Learning to diagnose with LSTM Recurrrent Neural Networks. International Conference on Learning Representations 2016. Retrieved from https://arxiv.org/abs/1511.03677.


