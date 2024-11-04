from copy import deepcopy
from queue import LifoQueue

def pauli_at(pauli: str, index: int, length: int):
    label = []
    for i in range(length):
        if (i + 1) == index:
            label.append(pauli)
        else:
            label.append("I")

    return "".join(label)

def enumerate_choices(n: int, r: int) -> list:
    choices = []
    stack = LifoQueue()
    stack.put([])

    while not stack.empty():
        item = stack.get()

        if len(item) == r:
            choices.append(item)
        else:
            start_idx = (item[-1] + 1) if len(item) > 0 else 0
            for i in range(start_idx, n):
                new_item = deepcopy(item)
                new_item.append(i)
                stack.put(new_item)

    return choices
