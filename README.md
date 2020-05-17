The repository contains scripts that help to convert Discourse Representation Structures (DRSs) into Graph formatted as JSON, the format adopted at this [meaning representation parsing shared task](http://mrp.nlpl.eu)   

### Convert DRSs in a CLF form into MRP graphs
`clf2graph.py` supports different types of conversion depending on how to represent concept or role clauses (labelled node vs labelled edge), where to place role nodes (between vs as a parent of arguments), and whether to label argument edges (with vs without ARG[12]).  

An example of converting `dev.txt` into `dev.bmall.mrp`, treating role and concept DRS conditions as labeled nodes, labeling role argument edges with `ARG1` and `ARG2`, and explicitly asserting box memberships for all discourse referents and reified roles (`-bm all`):

```
./clf2graph.py --input pmb-gold-split/clf/dev.txt --raw pmb-gold-split/clf/dev.txt.raw --output pmb-gold-split/mrp/dev.bmall.mrp --sig clf_signature.yaml # -bm all parameter is by default
```

Get simpler graphs where box-membership edges are often omitted for roles (see the definition of the parameters below):
```
./clf2graph.py --input pmb-gold-split/clf/dev.txt --raw pmb-gold-split/clf/dev.txt.raw --output pmb-gold-split/mrp/dev.rmid-bmarg1.mrp -rmid -bm arg1 --sig clf_signature.yaml
```

#### Supported conversion types
   - `-ce` treat concepts as labelled edges, otherwise as labelled nodes (default)  
   - `-rle` treat roles as unlabeled nodes with ingoing labelled edges, otherwise as labelled nodes (default)  
   - `-rmid` place roles between their arguments, otherwise as a parent of its arguments (default)  
   - `-noarg` don't place `ARGn` labelled or argument edges, otherwise place them (default)  
   - `-bm [all arg1 role a1]` box membership (`bm`) representation modes:
     - `all` `bm` edges are present;
     - `bm` edges are kept for `role`s but omitted for role arguments if both arguments (excepy constants) have the same membership as their role;
     - `arg1` `bm` edges are kept, but removed for its role if `arg1` has only that `bm` edge what its role has.
     - `a1` further develops `arg1` and also omits the `bm` edge for arg2 if the latter has `bm` edge for the same box as arg1 and teh role.
     -
### Requirements

Python 3
