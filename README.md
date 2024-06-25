# a tiny uuebassembly interpreter

2024 June 24:

up next:

- fix the Take<P>::advance function (it currently faults on a subtraction)
- finish type section
- break up the workspace into crates: nodes, parser, runtime
    - break the parser up into files
    - target sub-productions to make it easier to write tests
- look into injecting parser middleware
- ask ourselves: is there an easy way to support "read me a vec of this type"?



---



here's the plan:

- kind of big break

