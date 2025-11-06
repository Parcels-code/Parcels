# Documentation Notes

## Vision

We believe a clear documentation is important to community building, reproducibility, and transparency in our open-source project. To make it easier to write our documentation in a consistent way, here we outline a brief vision for our documentation based heavily on a few common resources.

```{note}
TODO: outline functions of the documentation based on resources
```

### Resources

- [Divio Documentation System](https://docs.divio.com/documentation-system/)
- [PyOpenSci Documentation Guide](https://www.pyopensci.org/python-package-guide/documentation/index.html#)
- [Write the Docs Guide](https://www.writethedocs.org/guide/)
- [NumPy Documentation Article](https://labs.quansight.org/blog/2020/03/documentation-as-a-way-to-build-community)

## Style guide

- **Write out `parcels.class.method` in tutorials and how-to guides** so that we can see which classes and methods are part of Parcels. If we use `from parcels import class`, the use of `class` in a cell further along is not obviously part of `parcels`.
- **Import packages at the top of the section in which they are first used** to show what they are used for.
- **Write documentation in first person plural ("we").** In our open source code, tutorials and guides can be written by any developer or user, so the documentation teaches all of us how to do something with Parcels. Sometimes it can be more natural to take on the tone of a teacher, writing to a student/learner, in which case it is okay to use "you". Please refrain from using impersonal subjects such as "the user".
