from collections import OrderedDict


class HTML2VECConverter:
    HTML2VEC_DIRECTION = 0
    VEC2HTML_DIRECTION = 1

    html_int_map = {
        '</': 0,
        '/>': 1,
        '<': 2,
        '>': 3,
        '=': 13,
        'body': 4,
        'input': 5,
        'table': 6,
        'div': 7,
        'p': 8,
        'button': 9,
        'class': 10,
        'head': 11,
        'html': 12,
        'li': 13,
        'ul': 14,
        'ol': 15,
        'tr': 16,
        'td': 17,
        'link': 18,
        'textarea': 19,
    }

    def __init__(self):
        self.html_int_map = OrderedDict(sorted(self.html_int_map.items(), key=lambda x: x[1]))

    def _clear_data(self, data):
        return data.replace('\n', '')

    def _get_next_item(self, data):
        origin_data = data[::]
        while data:
            for k, v in self.html_int_map.items():
                if data.startswith(k):
                    return k, data[len(k):]
            data = data[1:]
        return None, origin_data

    def split_html(self, data):
        result = []
        html = self._clear_data(data[::])
        while html:
            node, new_html = self._get_next_item(html)
            if node:
                result.append(node)
                html = new_html
        return result

    def convert(self, data, direction=HTML2VEC_DIRECTION):
        if direction == self.HTML2VEC_DIRECTION:
            result = [self.html_int_map[node] for node in self.split_html(data)]
        elif direction == self.VEC2HTML_DIRECTION:
            reversed_map = {v:k for k,v in self.html_int_map.items()}
            result = ''.join([reversed_map[num] for num in data])
        return result
