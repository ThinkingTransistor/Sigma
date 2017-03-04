# Sigma 
[![Build Status (Master)](https://img.shields.io/travis/ThinkingTransistor/Sigma/master.svg?style=flat-square)](https://travis-ci.org/ThinkingTransistor/Sigma)
[![Build Status (Development)](https://img.shields.io/travis/ThinkingTransistor/Sigma/development.svg?style=flat-square)](https://travis-ci.org/ThinkingTransistor/Sigma/branches)
[![MIT license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](http://choosealicense.com/licenses/mit)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg?style=flat-square)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=CQ2TPRV8Y6J9U)

Rocket powered machine learning. Create, compare, adapt, improve - neural networks at the speed of thought.

Short overview about why anyone would use this, how it came to be (even shorter) and who is supporting it for the future

## Installation

### NuGet

The recommended way to use the latest version of Sigma is to add the NuGet dependency to your project. Package follows soon...
<link>

### From source

For extensive customisation you can also install Sigma from source. This is not recommended as it may be outdated and unstable, but you still might want to do it for whatever reason. First, clone from the GitHub repository - use master for stable releases, development for recent and possibly unstable changes and fixes:

Restore and add all used NuGet packages (also see Used libraries):

You can then integrate Sigma as usual and compile using.

## Obligatory MNIST example
Very short first sample program to demonstrate capabilities, link to Samples project

## Documentation---How do I? 
The API-Documentation (of the master-branch) is always available at our [Github-Page](https://thinkingtransistor.github.io/Sigma/). If you want it locally available, clone the gh-pages branch.

How and what short overview, link to API and internal documentation

## Contribute
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg?style=flat-square)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=CQ2TPRV8Y6J9U)

Contribution guidelines, issue tracking, versioning (?), style requirements, also refer to internal documentation for backend modifications

## Used libraries

The following libraries / frameworks are used in the core project:

| Library                             | Purpose                           |
| :-----------------------------------|:----------------------------------|
| [log4net](https://logging.apache.org/log4net/) | Logging (log4j for .NET) |
| [NUnit](https://www.nunit.org/) | Unit testing |
| [DiffSharp](https://github.com/DiffSharp/DiffSharp), [Sigma.DiffSharp](https://github.com/GreekDictionary/Sigma.DiffSharp) | Functional automatic differentiation with ndarrays and various backends |
| [SharpZipLib](http://www.icsharpcode.net/) | Compression and decompression of various formats (zip, tar, gz, bzip, gzip) |
| [ManagedCuda](https://github.com/kunzmi/managedCuda), [ManagedCuda-CUBLAS](https://github.com/kunzmi/managedCuda) | Managed CUDA (GPU) and CuBLAS support |


The following libraries / frameworks are used in the WPF visualiser:

| Library                             | Purpose                           |
| :-----------------------------------|:----------------------------------|
| [Dragablz](https://github.com/ButchersBoy/Dragablz) | Tearable tab control for WPF, which includes docking, tool windows |
| [LiveCharts](https://github.com/beto-rodriguez/Live-Charts), [LiveCharts.Wpf](https://github.com/beto-rodriguez/Live-Charts) | Charting, graphing, advanced data, plotting |
| [MahApps.Metro](https://github.com/MahApps/MahApps.Metro), [MahApps.Metro.Resources](https://github.com/MahApps/MahApps.Metro) | Toolkit for creating metro-style WPF applications |
| [MaterialDesignColors](https://github.com/ButchersBoy/MaterialDesignInXamlToolkit), [MaterialDesignThemes](https://github.com/ButchersBoy/MaterialDesignInXamlToolkit), [MaterialDesignThemes.MahApps](https://github.com/ButchersBoy/MaterialDesignInXamlToolkit) | Material design for WPF/MahApps |


## Acknowledgements 

Special thanks to xyz
