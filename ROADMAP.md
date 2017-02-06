# Sigma Roadmap

The never ending list of nice-to-have things by project. These lists are not meant to be taken entirely seriously. Though they would be nice to have.

## Sigma.Core Roadmap

* Automated hyperparameter search across trainers
  * "Supervisor" to easily manipulate certain parameters
  * Multiple search modes for supervisor (e.g. basic grid search, advanced hyperparameter optimisation) 
* Truly distributed operator across devices
  * Operates on sub-operators for each device
* Plugin / layer dependency management
  * Automically load and inject custom layers
  * Custom code can be stored within the environment / trainer files 

## Sigma.Core.Monitors General Roadmap

* TCP monitor
  * allow TCP connections to receive and send hooks (just like any monitor)
  * remote monitoring and interaction from other devices
* HTTP monitor
  * on top of TCP monitor
  * as a website (maybe webhost) to easily remotely monitor and do fancy stuff
* Android / iOS monitor
  * on top of HTTP / TCP monitor
  * monitor only (no learning on phone)
  * learning on phone

## Sigma.Core.Monitors.WPF Roadmap

* Language change within the GUI having to restart _everything_
