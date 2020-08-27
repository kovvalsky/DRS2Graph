#!/usr/bin/env python3
#-*- coding: utf8 -*-

#################################
from os import path as op
import logging
from logging import debug, info, warning, error
import clf_referee as clfref
from collections import OrderedDict, defaultdict, Counter
import re
import os
import argparse
import json
import sys

#################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert mrp graphs into clauses")

    parser.add_argument(
    '--mrp', metavar='FILE PATH',
        help='File containing mrp graphs')
    parser.add_argument(
    '--clf', metavar='FILE PATH',
        help='File where clausal form will be written')
    parser.add_argument(
    '--sig', metavar = 'FILE PATH',
        help='If added, this contains a file with all allowed roles\
              otherwise a simple signature is used that\
              mainly recognizes operators based on their formatting')
    parser.add_argument(
    '--ids', nargs='*', metavar='LIST OF IDS',
        help='List of IDs of mrp which will be processed')

    parser.add_argument(
    '--validate', action="store_true",
        help="Validate with CLF referee")
    parser.add_argument(
    '--throw-error', action="store_true",
        help="Throw an error instead of counting them")
    parser.add_argument(
    '-v', dest="verbose", default=1, type=int, choices=[0, 1, 2], metavar="LEVEL",
        help="Verbosity of logging: warning(0), info(1), debug(2)")

    # pre-processing arguments
    args = parser.parse_args()
    # Set verbosity
    verbose = {0:logging.WARNING, 1:logging.INFO, 2:logging.DEBUG}
    logging.basicConfig(format='%(levelname)s: %(message)s', level=verbose[args.verbose])
    return args

def print_dict(d, pr=True):
    message = '\n'.join([ "{}: {}".format(k, d[k]) for k in sorted(d) ])
    if pr: print(message)
    return message

def find_disjunctions(edges):
    for (s1, t1), e1 in edges.items():
        if e1['lab'] == 'DIS':
            for (s2, t2), e2 in edges.items():
                if e2['lab'] == 'DIS':
                    if t1 == s2:
                        yield s1, t1, t2


def find_binary_pred_condition(node, edges, node_types, id=None):
    args, b, edge_num = [None, None], None, 0
    nid, _ = node
    for s_t in edges:
        if nid in s_t:
            edge_num += 1
            i = 1 - s_t.index(nid)
            if node_types[s_t[i]] == 'b':
                b = s_t[i] # box which is containg the predcate found
            else:
                args[i] = s_t[i] # argument found
    if not(2 <= edge_num <= 3):
        error("{}: Binary pred {} has many edges ({})".format(id, node, edge_num))
    if any([ j is None for j in args ]):
        error("{}: Binary pred {} is missing args ({})".format(id, node, args))
    if b is None:
        b = next((s for (s, t) in edges if t == args[0] and node_types.get(s, 0) == 'b'), None)
    if b is None:
        error("{}: can't find box of Binary pred {}".format(id, node))
    return b, args


#######################################
def mrp2clf(mrp, throw_error=False):
    # read nodes and edges and add None labels if they don't have any
    clf_info = {}
    node_types = {}
    nodes = { n['id']: {'lab':n.get('label', None)} for n in mrp['nodes'] }
    edges = { (e['source'], e['target']): {'lab':e.get('label', None)} for e in mrp['edges'] }
    id = mrp['id']
    debug(print_dict(nodes, pr=False))
    debug(print_dict(edges, pr=False))
    # Add typing info to edges and nodes. Don't use signature file at this stage
    # as many graphs can be dubbed invalid in the middle of conversion.
    # validation will be applied in the end when clf is generated

    for (s, t), e in edges.items():
        # Process Discourse edges except disjunctions
        if e['lab'] and e['lab'] not in ('in', 'DIS'):
            if not e['lab'].isupper():
                warning('{}: Suspicious Discourse connective: {}'.format(id, e['lab']))
            cl = 'b{} {} b{}'.format(s, e['lab'], t)
            clf_info[cl] = ('b', 'DRL', 'b')
            node_types[s] = node_types[t] = 'b' # using setdefault would be stricter
            e['done'] = nodes[s]['done'] = nodes[t]['done'] = True

    for b1, b2, b3 in find_disjunctions(edges):
        cl = 'b{} DIS b{} b{}'.format(b1, b2, b3)
        clf_info[cl] = ('b', 'DIS', 'b', 'b')
        node_types[b1] = node_types[b2] = node_types[b3] = 'b'
        edges[(b1, b2)]['done'] = edges[(b2, b3)]['done'] = True 

    for (s, t), e in edges.items():
        # Process in-edges
        if 'done' not in e and e['lab'] == 'in':
            # relax pattern to allow capturing even ill-formatted senses
            # some roles with explicit in-edge will leack here but their label will fail matching test
            lab = nodes[t]['lab']
            if lab is not None:
                m = re.match('(.+)\.([avnr]\.\d.+)', lab)
            if lab is None or m: # dealing with discourse referent
                cl2 = 'b{} REF x{}'.format(s, t)
                clf_info[cl2] = ('b', 'REF', 'x')
                node_types[s], node_types[t]  = 'b', 'x'
                e['done'] = nodes[s]['done'] = nodes[t]['done'] = True
            if lab is not None and m: # dealing with discourse referent with concept
                cl1 = 'b{} {} "{}" x{}'.format(s, *m.groups(), t)
                clf_info[cl1] = ('b', 'LEX', 'x')

    # process constant entities, which adds nothing to clauses yet
    for i, n in nodes.items():
        if n['lab'] and n['lab'][0] == n['lab'][-1] == '"':
            n['done'] = True
            node_types[i] = 'c'

    # add binary predicates that are applied to them: roles and operators
    # at this point they are the only unprocessed nodes
    for i, n in nodes.items():
        if 'done' not in n:
            b, args = find_binary_pred_condition((i, n), edges, node_types, id=id)
            # format differently constant and variable args
            term_args = [ 'x{}'.format(a) if node_types[a] == 'x' else nodes[a]['lab'] \
                            for a in args ]
            cl = 'b{} {} {} {}'.format(b, n['lab'], *term_args)
            clf_info[cl] = ('b', 'BIN', node_types[args[0]], node_types[args[1]])
            n['done'] = edges[(args[0], i)]['done'] = edges[(i, args[1])]['done'] = True
            if (b, i) in edges: edges[(b, i)]['done'] = True

    # check that all nodes and edges were processed
    undone_nodes = [ n for n in nodes.values() if 'done' not in n ]
    undone_edges = [ n for n in edges.values() if 'done' not in n ]
    if undone_nodes or undone_edges:
        error("{}: undone graph fragment: {}".format(id, undone_nodes + undone_edges))

    return clf_info

#######################################
def write_clfs(clf_infos, meta_list, filename=None):
    assert len(clf_infos) == len(meta_list)
    # open stream for writing, whether it is file or stdout
    if filename:
        OUT = open(filename, 'w')
    else:
        OUT = sys.stdout
    # start writing meta data and clfs
    for clf_info, meta in zip(clf_infos, meta_list):
        OUT.write("%%% {}\n%%% {}\n".format(*meta))
        for cl in sorted(clf_info):
            OUT.write(cl + '\n')
        OUT.write('\n')
    if filename:
        OUT.close()

#######################################################################
################################ Main  ################################
if __name__ == '__main__':
    args = parse_arguments()
    sig = clfref.get_signature(args.sig)
    with open(args.mrp) as F:
        mrps = [ json.loads(l) for l in F ]
    info("{} mrps read".format(len(mrps)))
    # converting mrps into clfs one-by-one
    error_counter = Counter()
    clfs_info_list, meta_list, invalids = [], [], []
    for mrp in mrps:
        if args.ids and mrp['id'] not in args.ids: continue
        meta_list.append((mrp['id'], mrp['input']))
        try:
            clf = mrp2clf(mrp)
            # if signature is
            if args.validate:
                clfref.check_clf(clf, sig)
            clfs_info_list.append(clf)
        except:
            if args.throw_error: raise
            error("{}: {}".format(mrp['id'], sys.exc_info()))
            error_counter.update([sys.exc_info()[1]])
            invalids.append(mrp['id'])
            clfs_info_list.append({'b nevermatching.n.01 x':('b', 'LEX', 'x')})
    write_clfs(clfs_info_list, meta_list, filename=args.clf)
    if error_counter:
        print("Frequencies of erros")
        for err, c in error_counter.most_common():
            print("{:>5}: {}".format(c, err))
    print("{} ({:.1f}%) mrp conversions failed: {}".format(\
        len(invalids), len(invalids)/len(mrps)*100, invalids))
