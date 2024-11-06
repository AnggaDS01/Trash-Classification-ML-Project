from pathlib import Path
from itertools import islice
import os

space =  '    '
branch = '│   '
tee =    '├── '
last =   '└── '

def tree(dir_path: Path, level: int=-1, limit_to_directories: bool=False,
         length_limit: int=3000, output_file: str=None):
    """Given a directory Path object print a visual tree structure and save to a file."""
    dir_path = Path(dir_path) # accept string coerceable to Path
    files = 0
    directories = 0
    output_lines = []  # List to store the output

    def inner(dir_path: Path, prefix: str='', level=-1):
        nonlocal files, directories
        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                line = prefix + pointer + path.name
                yield line
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix + extension, level=level-1)
            elif not limit_to_directories:
                line = prefix + pointer + path.name
                yield line
                files += 1

    # Generate output
    print(dir_path.name)
    output_lines.append(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        print(line)
        output_lines.append(line)
    if next(iterator, None):
        limit_msg = f'... length_limit, {length_limit}, reached, counted:'
        print(limit_msg)
        output_lines.append(limit_msg)
    summary = f'\n{directories} directories' + (f', {files} files' if files else '')
    print(summary)
    output_lines.append(summary)

    # Save output to a file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

if __name__ == '__main__':
    parent_path = Path(os.getcwd())
    name_parent_path = str(parent_path).split(os.path.sep)
    output_txt = 'tree_output.txt'  # File name for the output
    tree(parent_path.parent / name_parent_path[-1], level=4, output_file=output_txt)