#!/usr/bin/env python3
# -*- coding: utf8 -*-

'''Convert drs.clf into a drg in json format.
   It supporst two types of input/output:
   * PMB data directory as input & a directory with files for each split as output
   * A file with CLFs inside as input & a file with DRGs inside as output
   Usage:
    python3 clf2graph.py --lang en  --data-dir pmb-3.0.0/data/  --out-dir working --status gold  --splits dev_00:00 test_10:10 --sig clf_signature.yaml  -v 2
    python3 clf2graph.py --input drs.clf --raw raw_texts.txt --output drs.mrp --sig clf_signature.yaml -rmid -bm arg1
'''

#################################
from os import path as op
import logging
from logging import debug, info, warning, error
import clf_referee as clfref
from collections import OrderedDict, defaultdict, Counter
import re
import os
import sys
import math
from functools import reduce
from operator import mul
import argparse
from datetime import datetime
import json
import itertools as it
from nltk.corpus import wordnet as wn # to detect redundant copncepts



#################################
def parse_arguments():
    parser = argparse.ArgumentParser(description=\
    'Convert CLFs into JSON format for cross-framework meaning representation parsing')

    # Arguments covering directories and files
    parser.add_argument(
    '-l', '--lang', choices=['en', 'de'], metavar="LANG", # Only EN and DE are included in XFMR parsing task
        help="Language of the documents")
    parser.add_argument(
    '--data-dir', metavar="DIR_PATH",
        help="Path to the 'data' directory of the PMB release")
    parser.add_argument(
    '--out-dir', metavar="DIR_PATH",
        help="The directory where the extracted data will be placed")
    parser.add_argument(
    '--input', metavar="FILE_PATH",
        help="The file with clfs")
    parser.add_argument(
    '--raw', metavar="FILE_PATH",
        help="The file containing raw texts")
    parser.add_argument(
    '--raw-sep', metavar="STR", default="\n",
        help="Separator of documents/texts in raw input")
    parser.add_argument(
    '--output', metavar="FILE_PATH",
        help="The file where DRGs in mrp format will be written")
    parser.add_argument(
    '--alignment', metavar="FILE_PATH",
        help="Create a seperate file with alignments")
    parser.add_argument(
    '--anchored-out', metavar="FILE_PATH",
        help="Create a seperate file with graph and alignments together")
    parser.add_argument(
    '--id-guide', metavar="FILE_PATH",
        help="Path to mrp file the node ids of whoich current output shoudl follow")
    parser.add_argument(
    '--mappings', metavar="FILE_PATH",
        help="File that contains mappings of some IDs to help for matching")
    parser.add_argument(
    '--sig', default = '',
        help='If added, this contains a file with all allowed roles\
              otherwise a simple signature is used that\
              mainly recognizes operators based on their formatting')

    # Arguments covering filters about splits, parts, annotation layers and their statuses
    parser.add_argument(
    '--status', default='gold', choices=['gold', 'silver', 'bronze'],
        help="The annotation status of DRSs")
    parser.add_argument(
    '--splits', nargs="+", metavar="LIST OF SPLIT_NAME:PART_PATTERN",
        help="File names of splits with their corresponding python regex for parts. "
             "An example suitable for silver data: all_silver:..")

    # convesion parameters
    parser.add_argument(
    '--keep-refs', action="store_true",
        help="Don't remove redundant REF clauses")
    parser.add_argument(
    '--pmb2', action="store_true",
        help="Expect as an input the DRSa formatted in caluses as in the PMB release 2")
    parser.add_argument(
    '-ce', '--concept-edge', dest='ce', action='store_true',
        help="Treat concept assertion as a labeled node or labeled edge")
    parser.add_argument(
    '-bm', '--box-membership', dest='bm', default='all', choices=['all', 'a1', 'arg1', 'role'],
        help="Show box membership for all nodes, for both role args but not for a role (if possible), "
        "for arg1 but nor for role and arg2 (if possible), only for role but not for args (if possible)")
    parser.add_argument(
    '-noarg', '--no-role-arg', dest='noarg', action='store_true',
        help="Don't use ARG1 and ARG2 labels for the deges spanning between a role node and its arguments")
    parser.add_argument(
    '-rmid', '--role-as-midway', dest='rmid', action='store_true',
        help="Whether a role node is a parent or midway for its arguments")
    parser.add_argument(
    '-rle', '--role-as-edge', dest='rle', action='store_true',
        help="Model role predicates as labeled edge")

    # meta and config arguments
    parser.add_argument(
    '--ids', nargs="*", metavar="LIST OF INT",
        help="IDs (i.e., index starting from 0) of the CLFs that will be processed.\
              This works only for the clf file mode.")
    parser.add_argument(
    '--throw-error', action="store_true",
        help="Throw an error instead of counting them")
    parser.add_argument(
    '-v', dest="verbose", default=1, type=int, choices=[0, 1, 2], metavar="LEVEL",
        help="Verbosity of logging: warning(0), info(1), debug(2)")
    parser.add_argument(
    '--prov', default="PMB (3.0.0)", help="Provenance of the generated output")
    parser.add_argument(
    '--frwk', default="drg", help="Framework name to be injected in mrp")
    parser.add_argument(
    '--mrpv', default=1.0, type=float, help="mrp version to be injected in mrp")
    parser.add_argument(
    '--flvr', default=2, type=int, help="Flavor type to be injected in mrp")

    # pre-processing arguments
    args = parser.parse_args()
    # lossless conversion constraint
    if args.noarg and not args.rmid:
        raise RuntimeError('Role nodes as top without ARGn labeled egdes is a lossy format')
    # visually ad-hoc conversion
    if args.bm in ['arg1', 'a1'] and args.rle\
    or args.bm == 'role' and args.ce:
        raise RuntimeError('Role edges with optional role membership edges are incosistent. '
                           'So, are concept edges with optional arg membership edges.')
    # check that at least dir or file input/output mode is active
    dir_mode = all([args.lang, args.data_dir, args.out_dir])
    if not dir_mode:
        assert all([args.input, args.raw]), "File I/O mode is active"
    # Set verbosity
    verbose = {0:logging.WARNING, 1:logging.INFO, 2:logging.DEBUG}
    logging.basicConfig(format='%(levelname)s: %(message)s', level=verbose[args.verbose])
    # Set --splits if it is not set. Data is split into train/dev/test only for gold status
    if args.splits is None and dir_mode:
        split_lang_gold = {'en':['dev:.0', 'test:.1', 'train:.[2-9]'],
                           'de':['dev:.[01]', 'test:.[23]', 'train:.[4-9]']}
        args.splits = split_lang_gold[args.lang] if args.status == 'gold' else [args.status+':..']
    return args


#################################
def read_mappings(mapping_file):
    mappings = dict()
    with open(mapping_file) as MAP:
        for l in MAP:
            m = json.loads(l)
            assert len(m['from']) == len(m['to']), 'ill mapping'
            mappings[m['id']] = dict(zip(m['from'], m['to']))
    return mappings


#################################
def read_clfs(clf_file):
    '''Read clfs and token alignments from a file and return
       a list of CLFs, where each CLF is a pair of
       a list of clauses (i.e. tuples) and a list of (token, (start_offset, end_offset))
    '''
    info("reading " + clf_file)
    list_of_pd_clf_aligns = []
    clf, alignment = [], []
    # read CLFs where an empty line is a delimiter of CLFs
    with open(clf_file) as CLFS:
        for i, line in enumerate(CLFS, start=1):
            #print("{}:\t{}".format(i, line))
            # ignore commnet lines unless it contains pXX/dXXXX document ID
            if line.strip().startswith("%"):
                m = re.search('/(p\d{2}/d\d{4})/', line)
                if m:
                    pd = m.group(1)
                continue
            # empty line means an end of the running CLF
            if not line.strip():
                if clf:
                    assert len(clf) == len(alignment),\
                        "#clauses ({}) is the same as #alignments ({})".format(\
                        len(clf), len(alignment))
                    list_of_pd_clf_aligns.append((pd, clf, alignment))
                    clf, alignment = [], []
                continue
            # get a clause and a token alignment
            try: # when alignmnets are included
                clause, tok_offs = line.strip().split(' %', 1) # solves % % [38...39] case
            except ValueError: # when no alignments are included
                clause, tok_offs = line.strip(), None
            # get every aligned token for a clause
            cl_align = []
            # when there are alignments process them
            if tok_offs and tok_offs.find(' [') != -1:
                # treat also cases like: b3 REF e3   % ] [77...78]
                for tok_off in re.split('(?<=[0-9])\] ', tok_offs.strip()):
                    m = re.match('([^ ]+) \[(\d+)\.\.\.(\d+)', tok_off)
                    tok, start, end = m.groups()
                    cl_align.append((tok, (int(start), int(end))))
            # empty list is edded for each clause when there are no alignments
            alignment.append(cl_align)
            clause = tuple(clause.strip().split(' '))
            # clause should have 3 or 4 components
            if len(clause) in [3, 4]:
                clf.append(clause)
            else:
                warning('ill-formed clause: {}'.format(clause))
    # add the last clf which has no empty following it
    if clf:
        list_of_pd_clf_aligns.append((pd, clf, alignment))
    info("{} clfs read".format(len(list_of_pd_clf_aligns)))
    return list_of_pd_clf_aligns

#################################
def graph2mrp(id, raw, graph, features):
    '''Make a dictionary that will be dumped as json in mrp format
    '''
    framework = features.get('framework', 'drg')
    id_sorted_nodes = sorted(graph[0], key=lambda n: n['id'])
    src_trg_ordered_edges = sorted(graph[1], key=lambda e: (e['source'], e['target']))
    # format nodes and edges according to the framework
    if framework == 'drg':
        # get rid of anchors if not required
        if features.get('flavor', 2) == 1:
            keys = 'id label source target anchors'.split()
        else:
            keys = 'id label source target'.split()
        nodes = [ keep_keys(n, keys) for n in id_sorted_nodes ]
        edges = [ keep_keys(e, keys) for e in src_trg_ordered_edges ]
    elif framework == 'alignment':
        # get rid of labels and ?add ids to egdes
        keys = 'id source target anchors'.split()
        nodes = [ keep_keys(n, keys) for n in id_sorted_nodes ]
        edges = [ keep_keys(e, keys) for e in src_trg_ordered_edges ]
        for i, e in enumerate(edges):
            e['id'] = i
            e.move_to_end('id', last=False)
    else:
        raise ValueError('Unknown framework {}'.format(framework))
    # construct the final mrp
    item = [('id', id),
            ('flavor', features.get('flavor', 2)),
            ('framework', features.get('framework', 'drg')),
            ('version', features.get('version', 1.0)),
            ('time', datetime.now().strftime('%Y-%m-%d')),
            ('provenance', features.get('provenance', 'PMB (3.0.0)')),
            ('input', raw),
            ('nodes', nodes),
            ('edges', edges)
           ]
    if framework == 'drg':
        item = item[:7] + [('tops', graph[2])] + item[7:]
    return OrderedDict(item)

#################################
def keep_keys(d, keys):
    return OrderedDict([ (k, v) for k, v in d.items() if k in keys ])

#################################
def remove_quotes(s):
    '''Remove surrounding double-quotes'''
    assert set([s[0], s[-1]]) == set('"') and len(s) >= 3, "{} is quoted".format(s)
    return s[1:-1]

#################################
def connectivity_check(nodes, edges):
    '''Check the graph on connectivity (ignoring directions of edges)
    '''
    init = set()
    next = set([nodes[0]['id']])
    arcs = set([ (e['source'], e['target']) for e in edges ])
    while init != next:
        init = next.copy() # save old state
        used_arcs = set([ e for i in init for e in arcs if i == e[0] or i == e[1] ])
        for e in used_arcs: arcs.remove(e)
        for e in used_arcs: next.update([e[0], e[1]])
    diff = set([ n['id'] for n in nodes if n['id'] not in next ])
    assert not diff, "Nodes {} are disjoing from the rest".format(diff)

#################################
def remove_recoverable_edges(nodes, edges, bm):
    '''Remove box-membership edges if asked so
    '''
    if bm == 'all': return
    ty_nd = defaultdict(set)
    for n in nodes: ty_nd[n['type']].add(n['id'])
    ed = set([ (e['source'], e['target']) for e in edges ])
    # delete recoverable edges
    removables = [ rm for rm_set in recoverable_bm_edge(ty_nd, ed, bm) for rm in rm_set ]
    edges[:] = [ e for e in edges if (e['source'], e['target']) not in removables ]

#################################
def recoverable_bm_edge(nodes, edges, bm):
    for b in nodes['b']:
        for (s, rr) in edges:
            if s == b and rr in nodes['rr']:
                args = [ (x if rr == y else y) for (x, y) in edges if (x, y) != (s, rr) and rr in (x, y) ]
                assert len(args) == 2, "Role {} doesn't have two arguments".format(rr)
                if bm == 'role': # transfer box membership frpm role parent to arguments
                    # remove b-edges for args (not constants) if all have it
                    removables = set([ (b, a) for a in args if a in nodes['x']])
                else: # calculate bm-edges of each arg
                    a1 = args[0] if (args[0], rr) in edges else args[1] # detects arg1
                    # arg1 should have only b-edge, otherwise we cannot omit edges
                    if set([ bx for bx in nodes['b'] if (bx, a1) in edges ]) != set([b]):
                        continue
                    if bm == 'arg1':
                        removables = set([(b, rr)])
                    elif bm == 'a1': # induce mmembership to 2nd arg too
                        removables = set([ (b, a) for a in (args+[rr]) if a != a1 and a not in nodes['c'] ])
                    else:
                        raise RuntimeError("Don't know how to process bm={}".format(bm))
                if removables <= edges:
                    yield removables

#################################
def sanity_check_nodes(nodes):
    '''No duplicates for node IDs. Prefer labeled over unlabeled one (i.e., label=None)
       or specific concept label over general one.
       Also remove node properties that have None value
    '''
    id2lab = {}
    id2anch = defaultdict(list)
    for n in nodes:
        nid, nlab =  n['id'], n['label'] # note that all nodes have to have labels (can be None)
        if nid in id2lab: # this node was seen before
            # overwrite old node with a new labelled one if it is not None
            seen_lab = id2lab[nid]
            if seen_lab and nlab and seen_lab != nlab:
                raise RuntimeError("Different node labels from different boxes: {} and {}".format(\
                    seen_lab, nlab))
            else:
            # both labels are the same or one of them is None
                id2lab[nid] = nlab if nlab else seen_lab
        else:
            id2lab[nid] = nlab
        id2anch[nid] += n['anchors']
    # get set of nodes that follows the original node order
    ord_set_nodes = []
    for n in nodes:
        # if it has label or ultimately doesn't have lable, then include
        if n['id'] in id2lab:
            if id2lab[n['id']]: # has label != None
                n['label'] = id2lab[n['id']]
            else:
                del n['label']
            # keep or remove the anchors
            if id2anch[n['id']]: # has non empty anchors
                n['anchors'] = aligns2anchors(id2anch[n['id']])
            else:
                del n['anchors']
            ord_set_nodes.append(n)
            id2lab.pop(n['id'], None)
    return ord_set_nodes

#################################
def clean_set(multi_list):
    '''Make a list that will have unique elements preserving the order.
       Remove labels that are None
    '''
    ord_set = []
    for i in multi_list:
        if i not in ord_set:
            ord_set.append(i)
    # remove None labels and empty anchors
    for i in ord_set:
        if 'label' in i and i['label'] is None:
            del i['label']
        if 'anchors' in i and not i['anchors']:
            del i['anchors']
        else:
            i['anchors'] = aligns2anchors(i['anchors'])
    return ord_set

#################################
def aligns2anchors(aligns):
    set_aligns = sorted(set([ al[1] for al in aligns ]))
    return [ OrderedDict([('from', al[0]), ('to', al[1])]) for al in set_aligns ]


#################################
def add_edges(edges, eds, aligns=[]):
    '''Add tuple edges wrapped appropriately
    '''
    for ed, al in it.zip_longest(eds, aligns, fillvalue=[]):
        (s, t, l) = ed if len(ed) == 3 else (ed + (None,))
        assert isinstance(s, int) and isinstance(t, int), "Edge node IDs are integer"
        e = OrderedDict([('source', s), ('target', t)])
        if l: e['label'] = l
        e['anchors'] = al
        edges.append(e)

#################################
def iter_subseteq_iter(iter1, iter2):
    '''Iterator1 is subset or equal to iterator2 from a set perspective
    '''
    return set(iter1) <= set(iter2)

#################################
def add_nodes(nodes, nds, aligns=[]):
    '''Add tuple nodes wrapped appropriately
    '''
    for nd, al in it.zip_longest(nds, aligns, fillvalue=[]):
        (type, node_id, label) = nd if len(nd) == 3 else (nd + (None,))
        assert isinstance(node_id, int), "Node ID is integer"
        nodes.append(OrderedDict([('id', node_id), ('label', label), ('anchors', al), ('type', type)]))

#################################
def add_role(nodes, edges, cl, nid, role_id, aligns, pars={}):
    '''Augment a graph with a role, taking into account the conversion mode of role clauses
    '''
    (b, role, e, x) = cl
    # sanity check
    if pars['noarg'] and not pars['rmid']:
        raise RuntimeError('Role nodes as top without ARGn labeled egdes is a lossy format')
    # role clauses are always reified, and its label depends on the mode
    add_nodes(nodes, [('rr', role_id, None if pars['rle'] else role)], [aligns])
    # label of edge from box to role
    add_edges(edges, [ (nid[b], role_id, role if pars['rle'] else 'in') ])
    # arg labels on edges from role to args (if required)
    (arg1_lab, arg2_lab) = ('', '') if pars['noarg'] else ('ARG1', 'ARG2') #TODO symetric operators
    # adding edges depending on a mode
    if pars['rmid']:
    # role node is midway between its args
        add_edges(edges, [ (nid[e], role_id, arg1_lab), (role_id, nid[x], arg2_lab) ])
    else:
    # role node is a parent of its args
        add_edges(edges, [ (role_id, nid[e], arg1_lab), (role_id, nid[x], arg2_lab) ]) #TODO symetric operators
    # adding nodes depending on a mode


#################################
def clause_alignment(clf, cl_types, alignment):
    '''Create a mapping from box & clause to alignments
    '''
    cl2al = defaultdict(dict)
    for cl, ty, al in it.zip_longest(clf, cl_types, alignment):
        b = cl[0]
        if ty == 'REF': cl2al[b][cl[2]] = al
        elif ty == 'PRE': cl2al[(cl[0], cl[2])] = al
        elif ty == 'DRL': cl2al[(cl[1], cl[0], cl[2])] = al
        elif ty in ('LEX', 'ROL'): cl2al[b][cl[1:]] = al
        elif ty in ('DIS'): cl2al[b][cl[1:]] = al
        else: raise ValueError('Unknown type {}'.format(ty))
    return cl2al

#################################
def clf2graph(clf, alignment, signature=None, pars={}):
    '''Convert a CLF and alignments into a DRG graph
    '''
    # parse clf and check on correctness
    (box_dict, top_boxes, disc_rels, presupp_rels, cl_types, arg_typing) =\
        clfref.check_clf(clf, signature)
    assert len(clf) == len(cl_types), '#clauses == #clause_types'
    # map clauses to alignments
    cl2al = clause_alignment(clf, cl_types, alignment)
    # convert constants to nodes and get a mapping from terms to DIs
    nodes, nid = process_vars_constants(arg_typing)
    next_id = len(nid)
    # keep track of these
    edges = []
    # convert boxes into graph components
    for b, box in sorted(box_dict.items()):
        next_id = box2graph(box, nid, nodes, edges, next_id, arg_typing, cl2al, pars=pars)
    # add discourse relations
    for (r, b1, b2) in sorted(disc_rels):
        add_edges(edges, [(nid[b1], nid[b2], r)], [cl2al[(r, b1, b2)]])
    # add presupposition relations
    for (b1, b2) in sorted(presupp_rels):
        add_edges(edges, [(nid[b1], nid[b2], 'PRESUPPOSITION')], [cl2al[(b1, b2)]])
    # remove duplicate nodes but keep the order
    ord_set_nodes = sanity_check_nodes(nodes)
    if len(ord_set_nodes) != len(nodes):
        debug("After cleaning {} nodes remains {}".format(len(nodes), len(ord_set_nodes)))
    # roots = find_roots(ord_set_nodes, edges)
    edges = clean_set(edges)
    connectivity_check(ord_set_nodes, edges)
    remove_recoverable_edges(ord_set_nodes, edges, pars['bm'])
    # remove type feature from nodes, not needed anymore
    for nd in ord_set_nodes: del nd['type']
    debug("edges ({}); nodes ({})".format(len(edges), len(ord_set_nodes)))
    return ord_set_nodes, edges, [nid[b] for b in top_boxes]

#################################
def box2graph(box, nid, nodes, edges, next_id, arg_typing, cl2al, pars={}):
    b = box.name
    add_nodes(nodes, [('b', nid[b])]) # this adds all box nodes
    # extract concept conditions and referents
    concept_conds = [ c for c in box.conds if re.match('"[avnr]\.\d\d"$', c[1]) ]
    concept_ref = [ c[2] for c in concept_conds ]
    # process referents. This guarantees intro of every referent node in its box
    for x in sorted(box.refs):
        add_nodes(nodes, [('x', nid[x])], [cl2al[b][x]])
        if x not in concept_ref or pars['keep-refs']:
            # concept refs will be added when processing concept condition, important for -ce flag
            add_edges(edges, [(nid[b], nid[x], 'in')])
    for c in sorted(box.conds):
        (op, x, y) = c if len(c) == 3 else (c + (None,))
        # pmb2 version specific operators
        if len(c) == 2 and pars['pmb2']:
            assert op in ['NOT', 'POS', 'NEC'], "Unknown condition {} in pmb2 mode".format(c)
            add_edges(edges, [ (nid[b], op, nid[x]) ])
        elif op in ['IMP', 'DUP'] and pars['pmb2'] or op == 'DIS': # DIS p51/d2927 p25/d1536 p87/d2746
            # b --op1--> x --op2--> y
            # add_edges(edges, [ (nid[b], nid[x], op + '1'), (nid[x], nid[y], op + '2') ])
            # b --op--> x --op--> y
            add_edges(edges, [ (nid[b], nid[x], op), (nid[x], nid[y], op) ], [cl2al[b][c]])
        elif op in ['PRP'] and pars['pmb2']:
            # b --in--> x --PRP--> y
            add_edges(edges, [ (nid[b], nid[x], 'in'), (nid[x], nid[y], 'PRP') ])
        # pmb3 and pmb2 version
        elif c in concept_conds: # LEX case
            assert arg_typing[y] == 'x', "Concept is not applied to a referent: {}".format(c)
            # skip a hypernym concept condition
            hyp_cond = hyponym_condition(c, concept_conds)
            if hyp_cond: # transfer alignment of general concept to the specific one
                cl2al[b][hyp_cond] += cl2al[b][c]
            else:
                label = "{}.{}".format(op, remove_quotes(x))
                lab = None if pars['ce'] else label
                add_nodes(nodes, [('x', nid[y], lab)], [cl2al[b][(op, x, y)]])
                add_edges(edges, [ (nid[b], nid[y], label if pars['ce'] else 'in') ])
        elif op[0].isupper():
            # sanity check: covers roles like "PartOf" and EQU
            assert iter_subseteq_iter([arg_typing[i] for i in (x,y)], "xc")\
                   and (op[0:2].istitle() or op.isupper()), "Suspicious role: {}".format(c)
            add_role(nodes, edges, (b,) + c, nid, next_id, cl2al[b][c], pars=pars)
            next_id += 1
        else:
            raise RuntimeError("Cannot process condition {} in box {}".format(c, b))
    return next_id

#################################
def process_vars_constants(arg_typing):
    '''Get nodes for constants and a mapping from terms to IDs
    '''
    # Assign IDs to terms while preserving grouping IDs based on term types
    terms = sorted([a for a in arg_typing
                    if arg_typing[a] in 'bxc' and not re.match('"[avnr]\.\d\d"$', a)])
    t2id = { v: i for i, v in enumerate(terms) }
    # for constants already introduce labeled nodes
    const = set([a for a in terms if arg_typing[a] in 'c'])
    nodes = [ OrderedDict([('id', t2id[c]), ('label', c), ('anchors', []), ('type', 'c')])\
                for c in sorted(const) ]
    # replace referents and boxes with IDs in all box_dict
    return nodes, t2id

#################################
def hyponym_hypernym(ws1, ws2):
    '''Given two synset strings, check if ws1 is hyponym of ws2
    '''
    if ws2 == 'entity.n.01':
        return True
    patch_isa = set([('morning.n.01', 'time.n.08'), ('day.n.03', 'time.n.08'),
                     ('year.n.01', 'time.n.08'), ('century.n.01', 'time.n.08'),
                    ]) # dev:480, train:6182
    if (ws1, ws2) in patch_isa:
        return True
    # wordnet in the game
    ss1, ss2 = wn.synset(ws1), wn.synset(ws2)
    common = ss1.lowest_common_hypernyms(ss2)
    return common == [ss2]

#################################
def hyponym_condition(con_cond, conditions):
    '''Returns condition that has a hyponym concept of the given lexical condition
    '''
    (lex, sns, x) = con_cond
    ws = "{}.{}".format(lex, remove_quotes(sns))
    for c in conditions:
        if c != con_cond and x == c[2]:
            ws1 = "{}.{}".format(c[0], remove_quotes(c[1]))
            if hyponym_hypernym(ws1, ws):
                return c

#################################
def check_offsets(align, raw):
    '''Check that offsets really give the correct tokens'''
    # these are spans that boxer wrongly selects based on teh correct offsets
    problematic_spans = ('morning~after~pill', 'grownup', 'goalkeeper',
                   'Washington~D.~C.', 'Washington~DC', 'blowtorch', 'ballpoint~pen')
    for cl_align in align:
        for tok, (start, end) in cl_align:
            if tok in problematic_spans: continue
            assert tok == raw[start:end].replace(' ', '~'),\
                "Wrong offsets: {} != {}".format(tok, raw[start:end])

#################################
def read_guide_mrp(guide_mrp):
    '''Read guide MRP is a dictionary
    '''
    guide = dict()
    with open(guide_mrp) as F:
        for l in F:
            g = json.loads(l)
            guide[g['id']] = (g['nodes'], g['edges'])
    return guide

#################################
def guided_id_renaming(id, guide_mrp, graph, mappings={}):
    '''Adopt the node ids that are presented in guide mrp
    '''
    match_found = False
    nodes, edges, top = graph
    gnodes, gedges = guide_mrp[id]
    assert len(nodes) == len(gnodes), "{}: #nodes {} != {}".format(id, len(nodes), len(gnodes))
    assert len(edges) == len(gedges), "{}: #edges {} != {}".format(id, len(edges), len(gedges))
    # map nodes based on unique labels
    mapping = mappings[id] if id in mappings else {} # mapping from working to guide mrp
    uniq_id_labs, nuniq_lab_ids = nodes_uniq_rest(nodes, excl_ids=list(mapping.keys()))
    uniq_gid_glabs, nuniq_glab_gids = nodes_uniq_rest(gnodes, excl_ids=list(mapping.values()))
    assert len(uniq_id_labs) == len(uniq_gid_glabs),\
        "{}: diff num of once occuring: {} != {}".format(id, uniq_id_labs, uniq_gid_glabs)
    # create initial mapping absed on unque labels
    for i, l in uniq_id_labs:
        mapping[i] = next( gi for gi, gl in uniq_gid_glabs if gl == l )
    # do complete search untill mapping results full match
    n_poss = reduce(mul, [ math.factorial(len(v)) for v in nuniq_lab_ids.values() ], 1)
    if n_poss > 300000:
        print('Mapping search {}: {} possibilities'.format(id, n_poss))
    #return graph
    for mini_map in possible_mappings(nuniq_lab_ids, nuniq_glab_gids):
        #print('.', end='')
        full_map = {**mapping, **mini_map}
        assert len(full_map.values()) == len(full_map.keys()),\
            '{}: weak check on bijection'.format(id)
        if full_match(full_map, nodes, gnodes, edges, gedges):
            match_found = True
            break
    # apply full mapping to nodes and edges
    if not match_found:
        raise RuntimeError('Full match was not found')
    rnodes = [ OrderedDict(list(n.items()) + [('id', full_map[n['id']])]) for n in nodes ]
    redges = [ OrderedDict(list(e.items()) +
                [('source', full_map[e['source']]), ('target', full_map[e['target']])])
                for e in edges ]
    return rnodes, redges, top




#################################
def possible_mappings(lab_ids1, lab_ids2):
    permut_dict = dict()
    for lab, ids in lab_ids1.items():
        assert len(lab_ids2[lab]) == len(ids), "{} =! {}".format(lab_ids1, lab_ids2)
        permut_dict[lab] = list(it.permutations(ids))
    sorted_keys = sorted(lab_ids1.keys())
    ids2 = [ i for k in sorted_keys for i in lab_ids2[k] ]
    for permu in it.product(*[permut_dict[lab] for lab in sorted_keys]):
        per = [ i for l in permu for i in l ]
        yield dict(zip(per, ids2))


#################################
def full_match(full_map, nodes, gnodes, edges, gedges):
    nset = set( (full_map[n['id']], n['label'] if 'label' in n else None) for n in nodes )
    gnset = set( (n['id'], n['label'] if 'label' in n else None) for n in gnodes )
    if nset != gnset: return None
    eset = set( (full_map[e['source']], full_map[e['target']]) for e in edges )
    geset = set( (e['source'], e['target']) for e in gedges )
    if eset != geset: return None
    return True

#################################
def nodes_uniq_rest(nodes, excl_ids=[]):
    lab_ids = defaultdict(list)
    for n in nodes:
        if n['id'] in excl_ids: continue
        lab = n['label'] if 'label' in n else ''
        lab_ids[lab].append(n['id'])
    # ids and their unique labels
    uniq_id_labs = [ (ids[0], l) for l, ids in lab_ids.items() if len(ids) == 1  ]
    # non-unique labs and their associated id list
    nuniq_lab_ids = { l: ids for l, ids in lab_ids.items() if len(ids) != 1 }
    return uniq_id_labs, nuniq_lab_ids


#################################
def extract_splits_of_graphs(\
    data_dir, lang, status, splits, out_dir, sig=False, alignment=False, con_pars={}, throw_error=False):
    parts_dir = op.join(data_dir, lang, status)
    error_counter = []
    for split_pattern in splits:
        split_name, part_pattern = split_pattern.split(':', 1)
        # PMB parts that belong to a data split
        parts = [ p for p in os.listdir(parts_dir) if re.match('p'+part_pattern, p) ]
        # open file to write a split/partition of graphs
        with open(op.join(out_dir, split_name), 'w') as SPLIT:
            for p in parts:
                for d in os.listdir(op.join(parts_dir, p)):
                    # read clf and raw of a document
                    clf_file = op.join(parts_dir, p, d, lang+'.drs.clf')
                    raw_file = op.join(parts_dir, p, d, lang+'.raw')
                    if op.isfile(clf_file) and op.isfile(raw_file):
                        # only one clf is expected in a file under data
                        pd, clf, align = read_clfs(clf_file)[0]
                        with open(raw_file) as RAW:
                            raw = RAW.read().rstrip() # trailing white space only
                        if alignments: # check alignments if they are present
                            check_offsets(align, raw)
                        try:
                            graph = clf2graph(clf, align, signature=sig, pars=con_pars)
                        except:
                            error(sys.exc_info())
                            error_counter.append((p, d, sys.exc_info()[0]))
                            if throw_error: raise
                            continue
                        dict_drg = graph2mrp('p{}/d{}'.format(p, d[1:]), raw, graph, con_pars['prov'])
                        SPLIT.write(json.dumps(dict_drg) + '\n', ensure_ascii=False)
                    else:
                        warning("one of the files doesn't exist: {}, {}".format(clf_file, raw_file))
    return error_counter

##############################################################################
################################ Main function ################################
if __name__ == '__main__':
    args = parse_arguments()
    # read a signature file
    sig = clfref.get_signature(args.sig)
    # conversion parameters
    con_pars = {'pmb2':args.pmb2, 'keep-refs':args.keep_refs, 'ce':args.ce,
                'bm':args.bm, 'noarg':args.noarg, 'rmid':args.rmid, 'rle':args.rle,
                'prov':args.prov }
    # the directory I/O mode
    if all([args.lang, args.out_dir, args.data_dir]):
        if not op.exists(args.out_dir):
            os.makedirs(args.out_dir)
        error_counter = extract_splits_of_graphs(args.data_dir, args.lang, args.status,\
                            args.splits, args.out_dir, sig=sig, alignment=args.alignment,\
                            con_pars=con_pars, error=args.error)
    # the file I/O mode
    else:
        list_of_pd_clf_align = read_clfs(args.input)
        with open(args.raw) as r:
            list_of_raw = [ i for i in r.read().split(args.raw_sep) ]
            # if separator is trailing then discard the trailing empty raw
            if not list_of_raw[-1]: list_of_raw.pop()
        assert len(list_of_pd_clf_align) == len(list_of_raw), "Equal number of clfs and raws"
        error_counter = []
        # open files for writing
        if args.output:
            OUT = open(args.output, 'w')
        else:
            OUT = sys.stdout
        if args.alignment: OVERLAYS = open(args.alignment, 'w')
        if args.anchored_out: ANCH_MRP = open(args.anchored_out, 'w')
        # read guide mrp if specified
        if args.id_guide:
            guide_mrp = read_guide_mrp(args.id_guide)
        # read node mappings if provided
        mappings = read_mappings(args.mappings) if args.mappings else dict()
        # converting CLFs into Graphs one-by-one
        num = len(list_of_pd_clf_align) - 1
        for i, (pd, clf, align) in enumerate(list_of_pd_clf_align):
            if args.ids and not(str(i) in args.ids or pd in args.ids): continue
            raw = list_of_raw[i]
            #print(raw)
            if args.alignment:
                #check_offsets(align, raw)
                pass # due to boxer selecting wrong strings based on correct offsets
            try:
                graph = clf2graph(clf, align, signature=sig, pars=con_pars)
            except:
                error("ID {}: {}\n{}".format(pd, raw, sys.exc_info()))
                error_counter.append((i, sys.exc_info()[0]))
                if args.throw_error: raise
                continue
            # rename ids as in the guided mrp, if the latter is provided
            if args.id_guide:
                graph = guided_id_renaming(pd, guide_mrp, graph, mappings=mappings)
            # get main mrp graph
            feats = {'framework': args.frwk, 'flavor' :args.flvr,
                     'version': args.mrpv, 'provenance': args.prov }
            main_mrp = graph2mrp(pd, raw, graph, feats)
            if args.anchored_out:
                feats['flavor'] = 1
                anchored_mrp = graph2mrp(pd, raw, graph, feats)
                ANCH_MRP.write(json.dumps(anchored_mrp, ensure_ascii=False) + ('\n' if i < num else ''))
            #print(main_mrp)
            OUT.write(json.dumps(main_mrp, ensure_ascii=False) + ('\n' if i < num else ''))
            # get alignments
            if args.alignment:
                feats['framework'] = 'alignment'
                feats['flavor'] = 1
                feats['version'] = 1.1
                overlay_mrp = graph2mrp(pd, raw, graph, feats)
                OVERLAYS.write(json.dumps(overlay_mrp, ensure_ascii=False) + ('\n' if i < num else ''))
            #print(([ i['id'] for i in dict_drg['nodes']]))
            #print(([ (i['source'], i['target']) for i in dict_drg['edges']]))
        if args.output: OUT.close()
        if args.alignment: OVERLAYS.close()
        if args.anchored_out: ANCH_MRP.close()
    # print erros if any:
    print("Done.")
    if error_counter:
        print("{} errors".format(len(error_counter)))
        for i in error_counter:
            print(i)
