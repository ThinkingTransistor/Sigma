# Sigma 
[![Build Status (Master)](https://img.shields.io/travis/ThinkingTransistor/Sigma/master.svg?style=flat-square)](https://travis-ci.org/ThinkingTransistor/Sigma)
[![Build Status (Development)](https://img.shields.io/travis/ThinkingTransistor/Sigma/development.svg?style=flat-square)](https://travis-ci.org/ThinkingTransistor/Sigma/branches)
[![Nuget (PreRelease)](https://img.shields.io/nuget/vpre/Sigma.Core.svg?style=flat-square)](https://www.nuget.org/packages/Sigma.Core)
[![Nuget (PreRelease WPF)](https://img.shields.io/nuget/vpre/Sigma.Core.Monitors.WPF.svg?style=flat-square)](https://www.nuget.org/packages/Sigma.Core.Monitors.WPF)
[![MyGet (PreRelease)](https://img.shields.io/myget/sigma/v/Sigma.Core.svg?style=flat-square)](https://www.myget.org/feed/sigma/package/nuget/Sigma.Core)
[![MyGet (PreRelease WPF)](https://img.shields.io/myget/sigma/v/Sigma.Core.Monitors.WPF.svg?style=flat-square)](https://www.myget.org/feed/sigma/package/nuget/Sigma.Core.Monitors.WPF)
[![MIT license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](http://choosealicense.com/licenses/mit)

Rocket powered machine learning. Create, compare, adapt, improve - neural networks at the speed of thought.

Short overview about why anyone would use this, how it came to be (even shorter) and who is supporting it for the future

## Installation

### NuGet [Recommended]

The recommended way to use the latest version of Sigma is adding the NuGet package to your project. 
You can either include the core framework (command line only) [![Nuget (PreRelease)](https://img.shields.io/nuget/vpre/Sigma.Core.svg?style=flat-square)](https://www.nuget.org/packages/Sigma.Core) or the WPF visualiser (only works on windows) which also references the core framework [![Nuget (PreRelease WPF)](https://img.shields.io/nuget/vpre/Sigma.Core.Monitors.WPF.svg?style=flat-square)](https://www.nuget.org/packages/Sigma.Core.Monitors.WPF). 

In both cases, you can use any project with a main (ConsoleApplication) but you have to change the project settings to x64 since Sigma only supports 64bit mode.


### From source

For extensive customisation you can also install Sigma from source. This is not recommended as it may be outdated and unstable, but you still might want to do it for whatever reason. First, clone from the GitHub repository - use master for stable releases, development for recent and possibly unstable changes and fixes:

```
git clone https://github.com/ThinkingTransistor/Sigma
```

Restore and add all used NuGet packages (also see Used libraries) in the project folder (Sigma by default):

```
cd Sigma
nuget restore Sigma.sln
```

You can then integrate Sigma directly into your program as a project reference.

## First program - handwriting recognition with MNIST
Very short first sample program to demonstrate capabilities, link to Samples project

## Documentation - how do I? 
The API-Documentation (of the master-branch) is always available at our [Github-Page](https://thinkingtransistor.github.io/Sigma/). If you want it locally available, clone the gh-pages branch.

How and what short overview, link to API and internal documentation

## Contribute
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg?style=flat-square)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=CQ2TPRV8Y6J9U)

Contribution guidelines, issue tracking, versioning (?), style requirements, also refer to internal documentation for backend modifications

## Acknowledgements 

Special thanks to xyz

## Used libraries

For reference, a list of all libraries integrated with the Sigma. The following libraries / frameworks are used in the core:

| Library                             | Purpose                           |
| :-----------------------------------|:----------------------------------|
| [log4net](https://logging.apache.org/log4net/) | Logging (log4j for .NET) |
| [NUnit](https://www.nunit.org/) | Unit testing |
| [DiffSharp](https://github.com/DiffSharp/DiffSharp), [Sigma.DiffSharp](https://github.com/GreekDictionary/Sigma.DiffSharp) | Functional automatic differentiation with ndarrays and various backends |
| [SharpZipLib](http://www.icsharpcode.net/) | Compression and decompression of various formats (zip, tar, gz, bzip, gzip) |
| [ManagedCuda](https://github.com/kunzmi/managedCuda), [ManagedCuda-CUBLAS](https://github.com/kunzmi/managedCuda) | Managed CUDA (GPU) and CuBLAS support |


The following libraries are used in the graphical and interactive visualiser:

| Library                             | Purpose                           |
| :-----------------------------------|:----------------------------------|
| [Dragablz](https://github.com/ButchersBoy/Dragablz) | Tearable tab control for WPF, which includes docking, tool windows |
| [LiveCharts](https://github.com/beto-rodriguez/Live-Charts), [LiveCharts.Wpf](https://github.com/beto-rodriguez/Live-Charts) | Charting, graphing, advanced data, plotting |
| [MahApps.Metro](https://github.com/MahApps/MahApps.Metro), [MahApps.Metro.Resources](https://github.com/MahApps/MahApps.Metro) | Toolkit for creating metro-style WPF applications |
| [MaterialDesignColors](https://github.com/ButchersBoy/MaterialDesignInXamlToolkit), [MaterialDesignThemes](https://github.com/ButchersBoy/MaterialDesignInXamlToolkit), [MaterialDesignThemes.MahApps](https://github.com/ButchersBoy/MaterialDesignInXamlToolkit) | Material design for WPF/MahApps |
