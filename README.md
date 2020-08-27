The repository contains scripts that help to convert Discourse Representation Structures (DRSs) into Graph formatted as JSON, the format adopted at the [MRP shared task](http://mrp.nlpl.eu/2020), and back from the graph to DRS.     

### Convert DRSs from the Clausal Form (CLF) to Discourse Representation Graphs (DRG)
`clf2graph.py` supports different types of conversion depending on how to represent concept or role clauses (labelled node vs labelled edge), where to place role nodes (between vs as a parent of arguments), and whether to label argument edges (with vs without ARG[12]).  

#### Supported conversion parameters
   - `-ce` treat concepts as labelled edges, otherwise as labelled nodes (default)  
   - `-rle` treat roles as unlabeled nodes with ingoing labelled edges, otherwise as labelled nodes (default)  
   - `-rmid` place roles between their arguments, otherwise as a parent of its arguments (default)  
   - `-noarg` don't place `ARGn` labelled or argument edges, otherwise place them (default)  
   - `-bm [all arg1 role a1]` box membership (`bm`) representation modes:
     - `all` `bm` edges are present;
     - `bm` edges are kept for `role`s but omitted for role arguments if both arguments (excepy constants) have the same membership as their role;
     - `arg1` `bm` edges are kept, but removed for its role if `arg1` has only that `bm` edge what its role has.
     - `a1` further develops `arg1` and also omits the `bm` edge for arg2 if the latter has `bm` edge for the same box as arg1 and teh role.

#### CLF ↦ MRP DRG 
For example, the DRGs used in the [MRP shared task](http://mrp.nlpl.eu/2020/) are produced with (using a sample DRS): 
```
./clf2graph.py --input test_suite/crops/crops.clf --raw test_suite/crops/crops.raw --output test_suite/crops/crops.mrp -noarg -rmid -bm arg1 --sig clf_signature.yaml
```
Note that with this particular combination of parameters, some CLFs might not be convertible as the conversion assumes a single (maximally specific) concept per discourse referent.   

#### MRP DRG ↦ CLF
Clausal forms can be recovered from MRP DRGs by:
```
./mrp2clf.py --mrp test_suite/crops/crops.mrp --clf test_suite/crops/crops.mrp.clf --sig clf_signature.yaml
```

### Requirements
Python 3
