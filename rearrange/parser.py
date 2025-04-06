from typing import List, Set, Tuple, Union

class ParserError(Exception):
    pass

class PatternParser:
    def tokenize(self, pattern: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(pattern):
            if pattern[i].isspace():
                i += 1
            elif pattern[i] == '.' and i + 2 < len(pattern) and pattern[i:i+3] == '...':
                tokens.append('...')
                i += 3
            elif pattern[i] == '(':
                j = i + 1
                while j < len(pattern) and pattern[j] != ')':
                    j += 1
                if j == len(pattern):
                    raise ParserError("Unclosed parenthesis")
                group = pattern[i+1:j].split()
                tokens.append(tuple(group))
                i = j + 1
            elif pattern[i].isdigit():
                tokens.append(pattern[i])
                i += 1
            else:
                tokens.append(pattern[i])
                i += 1
        return tokens

    def parse(self, pattern: str) -> Tuple[List[Union[str, Tuple[str, ...]]], List[Union[str, Tuple[str, ...]]]]:
        parts = pattern.split('->')
        if len(parts) != 2:
            raise ParserError("Pattern must contain exactly one '->'")
        input_str, output_str = parts
        input_spec = self.tokenize(input_str.strip())
        output_spec = self.tokenize(output_str.strip())
        return input_spec, output_spec

    def get_axes(self, spec: List[Union[str, Tuple[str, ...]]]) -> Set[str]:
        axes = set()
        for item in spec:
            if isinstance(item, tuple):
                axes.update(item)
            elif item != '...' and item != '1':
                axes.add(item)
        return axes