import math
import argparse

# TODO: Add docstrings
# TODO: Proper comments
# TODO: Consider making DecTree class
target_attribute = ""
instances = []
attributes = []
tests = []

epsilon = 0.0001
output_file = "Output.dot"

# Utility functions
def split_instances(instances, attribute, value):
    subset = []
    rest = []
    for instance in instances:
        if instance.get_attr_val(attribute) == value:
            subset.append(instance)
        else:
            rest.append(instance)
    return (subset, rest)
    
def count_frequencies(instances, class_list, class_index_dict):
    frequencies = [0 for c in class_list]
    num_instances = len(instances)
    for instance in instances:
        frequencies[ class_index_dict[instance.get_classification()] ] += 1
    return frequencies

def sort_by_appearance(instances, attribute = None):
    vals = []
    vals_seen = {}
    i = 0
    for instance in instances:
        if attribute != None:
            val = instance.get_attr_val(attribute)
        else:
            val = instance.get_classification()
        if not val in vals_seen:
            vals_seen[val] = i
            i += 1
            vals.append(val)
    return (vals, vals_seen)

def print_node_data(node, node_id, printHist):
    global epsilon
    if node.is_split_node():
        color_attr = 'red'
        label = 'Split: ' + node.get_attribute()
    else:
        if node.get_entropy() < epsilon:
            color_attr = 'green'
        else:
            color_attr = 'blue'
        label = 'Leaf'

    num_instances = node.get_num_instances()

    s = '  ' + node_id + ' [shape=box, '
    s += 'color=' + color_attr + ', '
    s += 'label="' + label + '\\n'
    s += 'Entropy = ' + "{0:.4f}".format(node.get_entropy()) + '\\n'
    s += 'Instances = ' + str(num_instances) + '\\n'
    s += 'Decision = ' + node.get_classification()
    if printHist and num_instances > 0:
        s += '\\nHistogram:\\n'
        s += ' | { {val|num|prob}'
        for val, freq in node.get_frequencies():
            prob_str = "{0:.4f}".format(float(freq) / num_instances)
            s += ' |{' + str(val) + '|' + str(freq) + '|' + prob_str +'}'
        #s += ' }'
#        s += 'val\tnum\tprob\\n'
#        for val, freq in node.get_frequencies():
#            prob_str = "{0:.4f}".format(float(freq) / num_instances)
#            s += str(val) + '\t' + str(freq) + '\t' + prob_str +'\\n'
    s += '"] ;\r\n'
    return s

def print_edge_data(u_id, v_id, node_label):
    return '  ' + u_id + ' -> ' + v_id +  ' [label="' + node_label + '"] ;\r\n'

def postfix_traversal(root, node_id, printHist):
    num_nodes = 1
    num_splits = 0
    max_depth = 0

    s = print_node_data(root, node_id, printHist)
    z = ''

    child_nodes = root.get_children_list()
    if len(child_nodes) > 0:
        num_splits += 1
        for i in range(0, len(child_nodes)):
            attribute_val, child = child_nodes[i]
            child_id = node_id + "0" + str(i+1)
            s += print_edge_data(node_id, child_id, attribute_val)
            nnodes, nsplits, mdepth, output = postfix_traversal(child, child_id, printHist)
            num_nodes += nnodes
            num_splits += nsplits
            max_depth = max(max_depth, mdepth + 1)
            z += output

    return (num_nodes, num_splits, max_depth, s + '\r\n' + z)

class Instance:
    def __init__(self, val, attr_list, val_list):
        self.attrs = {}
        for i in range(0, len(attr_list)):
            self.attrs[attr_list[i]] = val_list[i]
        self.val = val

    def get_attr_val(self, attr_name):
        return self.attrs[attr_name]

    def get_classification(self):
        return self.val

class Node:
    def __init__(self):
        self.attribute = None
        self.classification = None
        self.frequencies = []
        self.children = []
        self.entropy = 0.0
        self.num_instances = 0

    def get_classification(self):
        return self.classification

    def set_classification(self, val):
        self.classification = val

    def get_entropy(self):
        return self.entropy

    def set_entropy(self, entropy):
        self.entropy = entropy

    def get_attribute(self):
        return self.attribute

    def set_attribute(self, attr_name):
        self.attribute = attr_name

    def get_num_instances(self):
        return self.num_instances

    def set_num_instances(self, num_instances):
        self.num_instances = num_instances

    def get_children_list(self):
        return self.children

    def add_child(self, node, attr_val):
        self.children.append( (attr_val, node) )

    def get_child(self, attr_val):
        for child in self.children:
            if child[0] == attr_val:
                return child[1]
        return None

    def is_split_node(self):
        return self.attribute != None

    def get_frequencies(self):
        return self.frequencies

    def set_frequencies(self, freq):
        self.frequencies = freq


def find_entropy(instances, class_list, class_index_dict):
    result = 0.0
    num_instances = len(instances)
    frequencies = count_frequencies(instances, class_list, class_index_dict)
    for partial in frequencies:
        coeff = partial / float(num_instances)
        if coeff:
            result -= coeff * math.log(coeff, 2)
    return result

def find_gain(instances, attribute_name, class_list, class_index_dict):
    entropy_s = find_entropy(instances, class_list, class_index_dict)
    _, attr_value_index_dict = sort_by_appearance(instances, attribute_name)

    partial_results = []
    for i in range(0, len(attr_value_index_dict.keys())):
        partial_results.append([])

    num_instances = len(instances)
    for instance in instances:
        val_index = attr_value_index_dict[instance.get_attr_val(attribute_name)]
        partial_results[ val_index ].append(instance)

    result = entropy_s
    for partial in partial_results:
        if len(partial) > 0:
            entropy_sv = find_entropy(partial, class_list, class_index_dict)
            result -= len(partial) / float(num_instances) * entropy_sv

    return result

def find_most_feasible_attribute(instances, attrs, class_list, class_indices):
    result = []
    for i in range(0, len(attrs)):
        gain = find_gain(instances, attrs[i], class_list, class_indices)
        result.append((gain, -i, attrs[i]))
    result.sort()
    return result[-1][-1]

def ID3(instances, target_attr, attributes):
    class_list, class_index_dict = sort_by_appearance(instances)

    root = Node()
    root.set_entropy(find_entropy(instances, class_list, class_index_dict))
    num_instances = len(instances)
    root.set_num_instances(num_instances)
    frequencies = count_frequencies(instances, class_list, class_index_dict)
    root.set_frequencies([(class_list[i], frequencies[i]) for i in range(0, len(frequencies))])

    most_frequent_class_index = 0
    for i in range(0, len(frequencies)):
        if frequencies[i] == num_instances:
            root.set_classification(class_list[i])
            return root
        if frequencies[most_frequent_class_index] < frequencies[i]:
            most_frequent_class_index = i

    root.set_classification(class_list[most_frequent_class_index])
    if len(attributes) == 0:
        return root

    best_attribute = find_most_feasible_attribute(instances, attributes,
                                                  class_list, class_index_dict)
    root.set_attribute(best_attribute)

    best_attr_vals, _ = sort_by_appearance(instances, best_attribute)
    for val in best_attr_vals:
        subset, instances = split_instances(instances, best_attribute, val)
        if len(subset) > 0:
            new_attrs = []
            for attr in attributes:
                if attr != best_attribute:
                    new_attrs.append(attr)
            node = ID3(subset, target_attr, new_attrs)
        else:
            node = Node()
            # Default/Initial values for entropy, frequencies and num_instances
            # are valid for node, which is why they need not be set.
            node.set_classification(class_list[most_frequent_class_index])
        root.add_child(node, val)
    return root

def run_tests(root):
    global tests

    f = open('log.txt', 'w')
    if len(tests) > 0:
        correct_guesses = 0
        for test in tests:
            node = root
            while node.is_split_node():
                node = node.get_child(test.get_attr_val(node.get_attribute()))
            guess = node.get_classification()
            if guess == test.get_classification():
                correct_guesses += 1
        accuracy = correct_guesses / float(len(tests))
        f.write('Test Accuracy: ' + "{0:.4f}".format(accuracy) + '\r\n')
    else:
        f.write('No test instances found.\r\n')
    f.close()

def read_input(input_file):
    global instances, attributes, target_attribute, tests
    attr_list = []
    train_list = []
    test_list = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            info = [elem.strip() for elem in line[2:-1].split(',')]
            if line.startswith('T:'):
                attr_list = info
            elif line.startswith('A:'):
                train_list.append(info)
            elif line.startswith('B:'):
                test_list.append(info)

    target_attribute = attr_list[-1]
    attributes = attr_list[:-1]

    for train_example in train_list:
        instance = Instance(train_example[-1], attributes, train_example[:-1])
        instances.append(instance)

    for test_example in test_list:
        test = Instance(test_example[-1], attributes, test_example[:-1])
        tests.append(test)

def print_output(root, printHist = False):
    global output_file
    nodes, splits, depth, details = postfix_traversal(root, "1", printHist)

    f = open(output_file, 'w')
    f.write("digraph G\r\n{\r\n  ")
    f.write('graph [label="Decision Tree\\n\\nNumber of Nodes = ' + str(nodes))
    f.write('\\nNumber of Splits = ' + str(splits) + '\\nMaximum Depth = ')
    f.write(str(depth) + '\\n\\n", labelloc=t] ;\r\n\r\n')
    f.write(details[:-2])
    f.write("}\r\n")
    f.close()

def print_output_with_histogram(root):
    print_output(root, True)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input', metavar='F', type=str, default="Input.txt",
                                 help='input file path')
    parser.add_argument('--hist', dest='print_output', action='store_const',
                                 const=print_output_with_histogram,
                                 default=print_output,
                                 help='print histogram for each node')
    args = parser.parse_args()

    read_input(args.input)
    root = ID3(instances, target_attribute, attributes)
    accuracy = run_tests(root)
    args.print_output(root)

if __name__ == '__main__':
    main()
