# Parcels documentation

Welcome to the documentation of Parcels. **Parcels** provides a set of Python classes and methods to create customisable particle tracking simulations using gridded output from (ocean) circulation models. Parcels can be used to track passive and active particulates such as water, plankton, [plastic](http://www.topios.org/) and [fish](https://github.com/Jacketless/IKAMOANA).

```{figure} _static/homepage.gif
:class: dark-light
```

_Animation of virtual particles carried by ocean surface flow in the global oceans. The particles are advected with Parcels in data from the_ [NEMO Ocean Model](https://www.nemo-ocean.eu/).

```{note}
You can browse the documentation for older versions by using the version switcher in the bottom right.
```

**Useful links**: [Installation instructions](getting_started/installation.md) | [Discussions on GitHub](https://github.com/Parcels-code/parcels/discussions) | [Issue on GitHub](https://github.com/Parcels-code/parcels/issues) | [Parcels website](https://parcels-code.org/) | [CLAM community website](https://clam-community.github.io/) | [API reference](reference/parcels/index)

`````{grid} 1 2 2 2
:gutter: 4
:padding: 2 2 0 0
:class-container: sd-text-center

````{grid-item-card} Getting started
:shadow: md

New to **Parcels**? Check out the installation guide, run the quickstart tutorial, and learn the key concepts to understand the package.

+++

```{button-ref} getting_started/index
:ref-type: doc
:click-parent:
:color: secondary
:expand:

Get started!
```
````
````{grid-item-card} How to?
:shadow: md

Wondering how to load a `FieldSet` or write a `Kernel`? Find **tutorials** and explainers to these and other questions here:

+++

```{button-ref} user_guide/index
:ref-type: doc
:click-parent:
:color: secondary
:expand:

To the user guide
```
````
````{grid-item-card} Development
:shadow: md

We encourage anyone to help improve **Parcels**: read our guidelines to get started!

+++

```{button-ref} development/index
:ref-type: doc
:click-parent:
:color: secondary
:expand:

Contributing guidelines
```
````
````{grid-item-card} Community
:shadow: md

Want to interact with other users and **Parcels** developers?

+++

```{button-ref} community/index
:ref-type: doc
:click-parent:
:color: secondary
:expand:

Connect with our community!
```
````
`````

```{toctree}
:maxdepth: 2
:hidden:

Home <self>
Getting started <getting_started/index>
User guide <user_guide/index>
Community <community/index>
Development <development/index>
API reference <reference/parcels/index>
v4 <v4/index>
Parcels website <https://parcels-code.org/>
```
