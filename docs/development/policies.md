# Policies

## Use of AI in development

Many developers use AI (and in particular Large Language Models) to help them in their work. These LLMs have received both praise and criticism when it comes to software development.

We accept that Parcels developers have their own motivation for using (or not using) AI. However, we have one policy that we expect all Parcels developers to follow:

> It is ultimately your responsibility to understand the code that you commit.

Remember that reviews are done by human maintainers - asking us to review code that an AI wrote but you don't understand isn't kind to these maintainers.

Furthermore, we also have an [AI agents policy](/CLAUDE.md) that states how we expect AI agents to engage with Parcels on GitHub.

## Versioning

Parcels follows [Intended Effort Versioning (EffVer)](https://jacobtomlinson.dev/effver/), where the version number (e.g., v2.1.0) is thought of as `MACRO.MESO.MICRO`.

> MACRO version - you will need to dedicate time to upgrading to this version<br>
> MESO version - some small effort may be required for you to upgrade to this version<br>
> MICRO version - no effort is intended for you to upgrade to this version<br>

When making backward incompatible changes, we will make sure these changes and instructions to upgrade are communicated to the user via change logs or migration guides, and (where applicable) informative error messaging.

Note when conducting research we highly recommend documenting which version of Parcels (and other packages) you are using. This can be as easy as doing `conda env export > environment.yml` alongside your project code. The Parcels version used to generate an output file is also stored as metadata entry in the `.zarr` output file.

## Changes in policies

- In v4.0.0 of Parcels, adopted EffVer which formalises this "SemVer-like" variant we were following - and we adjusted our deprecation policy.
- In [v3.1.0](https://docs.parcels-code.org/en/v3.1.0/community/policies.html) of Parcels, we adopted SemVer-like versioning and deprecation policies
