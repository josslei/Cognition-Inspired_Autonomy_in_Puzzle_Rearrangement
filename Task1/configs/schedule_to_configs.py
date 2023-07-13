import os
import argparse
import copy
from typing import Tuple, List, Dict, Any

import csv
import tomli
import tomli_w
from treelib import Tree, Node

TOML_FILE_PREFIX: str = 'scheduled_'

NUM_CUDA_DEVICE: int = 2

SCRIPT_DECLARATION: str = '#!/bin/bash\n'
RUN_COMMAND_PREREQUISITE: str = 'conda activate targf\n'
RUN_COMMAND_CUDA_PREFIX: str = 'CUDA_VISIBLE_DEVICES='
RUN_COMMAND_MAJOR: str = 'python3 main.py'

parser = argparse.ArgumentParser(description='Read schedule, generate configs & bash script(s).')

parser.add_argument('--path_schedule', type=str, required=True, help='path to the schedule file (.csv)')
parser.add_argument('-o', '--path_output', type=str, required=True, help='path to store output(s)')
parser.add_argument('--path_config_template', type=str, required=True, help='path to the config template (.toml)')

def read_schedule(path_schedule: str) -> Dict[str, List[Any]]:
    schedule: Dict[str, List[Any]] = dict()
    with open(path_schedule, 'r') as fp_schedule:
        csv_reader = csv.reader(fp_schedule)
        rows = [row for row in csv_reader]
    for i, key in enumerate(rows[0]):
        schedule[key] = []
        for j in range(1, len(rows)):
            if rows[j][i] == '':
                continue
            schedule[key] += [eval(rows[j][i])]
    return schedule


def schedule_to_tree(schedule_dict: Dict[str, List[Any]]) -> Tree:
    # Init tree
    schedule_tree: Tree = Tree()
    schedule_tree.create_node(tag='root', identifier='root', data=None)

    parent_list: List[Node] = [schedule_tree['root']]
    new_parent_list: List[Node] = []
    # Iterate settings
    for key in schedule_dict.keys():
        # Same under every current leaf
        for parent in parent_list:
            # For every branch
            for data in schedule_dict[key]:
                schedule_tree.create_node(tag=key, parent=parent, data=data)
            new_parent_list += schedule_tree.children(parent.identifier)
        # Update list of leaf nodes
        parent_list = new_parent_list
        new_parent_list = []

    #schedule_tree.show()
    #print(len(schedule_tree.leaves()))
    return schedule_tree


def tree_to_toml_and_script(save_path: str,
                            schedule_tree: Tree,
                            config_template: Dict[str, Any]) -> None:
    leader_list: List[Node] = schedule_tree.children('root')
    for index, leader in enumerate(leader_list):
        path_list, config_list = __paths_to_leaves(tree=schedule_tree, 
                                                   node=leader, 
                                                   index=index, 
                                                   path='', 
                                                   config=copy.deepcopy(config_template))

        # Write TOML config files
        os.system(f'mkdir -p {save_path}')
        for i, path in enumerate(path_list):
            toml_file_name: str = os.path.join(save_path, TOML_FILE_PREFIX + path + '.toml')
            with open(toml_file_name, 'wb') as fp_toml:
                config_list[i]['config_name'] = TOML_FILE_PREFIX + path
                tomli_w.dump(config_list[i], fp_toml)
        # Generate script
        script_file_name: str = os.path.join(save_path, f'run_script_{__index_to_letter(index)}.sh')
        with open(script_file_name, 'w') as fp_script:
            fp_script.write(SCRIPT_DECLARATION)
            fp_script.write('\n')
            fp_script.write(RUN_COMMAND_PREREQUISITE)
            for path in path_list:
                toml_file_name: str = os.path.join(save_path, TOML_FILE_PREFIX + path + '.toml')
                command: str = RUN_COMMAND_CUDA_PREFIX + str(index % NUM_CUDA_DEVICE) + ' '
                command += RUN_COMMAND_MAJOR + ' '
                command += toml_file_name + '\n'
                fp_script.write(command)
        pass


def __index_to_letter(index: int) -> str:
    return chr(0x41 + index)


def __paths_to_leaves(tree: Tree,
                      node: Node,
                      index: int,
                      path: str,
                      config: Dict[str, Any]) -> Tuple[List[str], List[dict]]:
    def __modify_toml_dict(config: Dict[str, Any], raw_key: str, value: Any) -> None:
        for key in raw_key.split('.')[:-1]:
            config = config[key]
        config[raw_key.split('.')[-1]] = value
    
    if node.is_leaf():
        path += __index_to_letter(index)
        __modify_toml_dict(config, str(node.tag), node.data)
        return [path], [config]
    else:
        path += __index_to_letter(index)
        __modify_toml_dict(config, str(node.tag), node.data)
        # Walk down
        path_list: List[str] = []
        config_list: List[dict] = []
        for i, child in enumerate(tree.children(node.identifier)):
            _path_list, _config_list = __paths_to_leaves(tree, child, i, path, copy.deepcopy(config))
            path_list += _path_list
            config_list += _config_list
        return path_list, config_list

if __name__ == '__main__':
    args = parser.parse_args()
    path_schedule: str = args.path_schedule
    path_output: str = args.path_output
    path_config_template: str = args.path_config_template

    schedule_dict: Dict[str, List[Any]] = read_schedule(path_schedule)

    schedule_tree: Tree = schedule_to_tree(schedule_dict)

    with open(path_config_template, 'rb') as fp_config_template:
        config_template: Dict[str, Any] = tomli.load(fp_config_template)
    tree_to_toml_and_script(path_output, schedule_tree, config_template)
