# Sigma Roadmap

The never ending list of nice-to-have things by project. These lists are not meant to be taken entirely seriously. Though they would be nice to have.

## Sigma.Core

* Automated hyperparameter search across trainers
  * "Supervisor" to automagically and easily manipulate certain parameters (with smart guesses)
  * Multiple search modes for supervisor (e.g. basic grid search, advanced hyperparameter optimisation) 
* Truly distributed operator across devices
  * Operates on sub-operators for each device
* Plugin / layer dependency management
  * Automically load and inject custom layers
  * Custom code can be stored within the environment / trainer files 
* Improved preprocessor pipeline with extra step before extraction
  * Use extra step to detect values for preprocessing (e.g. for auto-normalisation)

## Sigma.Core.Monitors General

* TCP monitor
  * Allow TCP connections to receive and send hooks (just like any monitor)
  * Remote monitoring and interaction from other devices
* HTTP monitor
  * On top of TCP monitor
  * As a website (maybe webhost) to easily remotely monitor and do fancy stuff
* Android / iOS monitor
  * On top of HTTP / TCP monitor
  * Monitor only (no learning on phone)
  * Learning (visualisation) on phone
* [Smonity](https://smonity.github.io/) monitor (SMonitor)
  * Monitor RAM
  * Receive notifications

## Sigma.Core.Monitors.WPF

* Language change within the GUI without having to restart _everything_
