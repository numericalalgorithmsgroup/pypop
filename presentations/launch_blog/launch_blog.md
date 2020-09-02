# PyPOP - A new tool for efficient performance analysis

We are proud to announce the launch of a new tool - PyPOP - designed to support the existing POP
tools with high quality analysis, plotting and report generation functionality.  This blog post
will explore the reasons why we wrote a new tool and give an overview of the functionality.

## Why we wrote another POP Tool

There are various profiling tools that are developed by partners in the POP project, including the
[Extrae], [Scalasca] and [Maqao] tools that have been discussed in [previous] [blog] [posts].
These are extremely powerful tools that can provide huge amounts of detailed information about
application behaviour, however, sometimes the sheer volume of information can be overwhelming both
for the analyst trying to interpret it and also in terms of the size of the trace files being
unable to fit in the memory of a laptop or workstation.

[Extrae]: https://tools.bsc.es/extrae
[Scalasca]: https://www.scalasca.org/
[Maqao]: http://www.maqao.org/
[previous]: https://pop-coe.eu/blog/pop-tool-descriptions-bsc-performance-tools
[blog]: https://pop-coe.eu/blog/pop-tool-descriptions-jsc-performance-tools
[posts]: https://pop-coe.eu/blog/pop-tool-descriptions-uvsq-performance-tools

The [POP methodology] has been developed in response to the difficulties involved in doing
performance analysis efficiently and effectively, and the [POP Metrics] are at the centre of this.
The idea of the POP methodology is to break down the application performance into different areas
such as parallel communication, use of hardware resources etc, each of which is reflected by one of
the POP metrics.  This high-level view informs the next stage of the analysis which can then target
the specific areas with poor performance in an iterative fashion until the root cause(s) of any
inefficiencies are identified and fixed.

[POP Methodology]: https://pop-coe.eu/further-information/online-training/understanding-application-performance-with-the-pop-metrics
[POP Metrics]: https://pop-coe.eu/node/69

The development of PyPOP came about as a natural result of the POP Methodology: we needed a tool
that allowed us to efficiently and repeatably analyze profiling results and compute the POP
metrics.  As the tool matured it gained additional features such as plotting capabilities, batch
pre-processing of large traces on remote HPC nodes and user interface features to improve ease of
use.  

## PyPOP Features

PyPOP has been designed to allow efficient, repeatable performance analysis. It re-uses the
available features of the underlying profiling tools where possible, and is written in Python3 with
the widely used Numpy and Pandas libraries to minimize barrier-to-entry for custom analysis.

#### Automated calcuation of the POP Metrics

Trace files are directly supplied as input. PyPOP then processes them to extract the relevant data
and calculate the POP Metrics.  As well the original POP MPI Metrics various other metric types
in development by the project can be calculated.

#### Jupyter notebook interface with interactive "Wizard"

PyPOP uses a "literate programming" approach that can mix GUI elements, text, figures and Python
code within the Jupyter "electronic notebook" environment.  This gives great flexibility to use
PyPOP in either a GUI-style "wizard mode" or to use the full power of the Python language to
rapidly develop custom analyses on top of the PyPOP framework.

#### High quality table and scaling plot generation

PyPOP provides routines to plot both formatted metrics tables and scaling plots for parallel
speedup and other metrics. The produced plots are interactive, with mouseover details and
descriptions and can be saved as images for later use.

#### PDF report generation

Once the metrics have been calculated and the plots produced, PyPOP supports creation of PDF
formatted reports directly from the Jupyter Notebook.


#### Headless pre-analysis mode

PyPOP also has a command-line interface which can be used to pre-process very large (10GB or more)
traces on remote machines such as HPC nodes or cloud instances. This can be done as a batch job for
example, and the small (less than 100kB) summary file downloaded to analyse locally using the PyPOP
GUI interface.


## Getting Started with PyPOP

To download a copy of PyPOP visit the [PyPOP Github repository]. The [Readme] and [Documentation]
describe how to install PyPOP along with example analysis files to use with the tutorial.

We also have an [Introduction to PyPOP] module part of the [POP Online Training Course].

<p><iframe allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen="" frameborder="0" height="315"
src="https://www.youtube.com/embed/eJ4SfkycI4A?list=PLDPdSvR_5-GgOV7MDtvP2pzL29RRrMUqn;rel=0"
width="560"></iframe></p>

[PyPOP Github repository]: https://github.com/numericalalgorithmsgroup/pypop/
[Readme]: https://github.com/numericalalgorithmsgroup/pypop/
[Documentation]: https://numericalalgorithmsgroup.github.io/pypop/doc.html

[Introduction to PyPOP]: https://pop-coe.eu/further-information/online-training/computing-the-pop-metrics-with-pypop
[POP Online Training Course]: https://pop-coe.eu/further-information/online-training
