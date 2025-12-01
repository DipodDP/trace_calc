import logging

class TruncatingFilter(logging.Filter):
    def __init__(self, name: str = "", max_length: int = 250):
        super().__init__(name)
        self.max_length = max_length

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:  # Check if arguments are present (typically for format strings)
            new_args = []
            for arg in record.args:
                if isinstance(arg, str) and len(arg) > self.max_length:
                    new_args.append(arg[:self.max_length] + "...")
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)
        elif isinstance(record.msg, str) and len(record.msg) > self.max_length:
            # If no args, and it's a string, truncate the message itself (f-string or literal)
            record.msg = record.msg[:self.max_length] + "..."
        return True
