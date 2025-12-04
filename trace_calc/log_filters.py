import logging

class TruncatingFilter(logging.Filter):
    def __init__(self, name: str = "", max_length: int = 250):
        super().__init__(name)
        self.max_length = max_length

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:  # Check if arguments are present
            new_args = []
            for arg in record.args:
                s_arg = str(arg)  # Get string representation
                if len(s_arg) > self.max_length:
                    new_args.append(s_arg[:self.max_length] + "...")
                else:
                    new_args.append(arg)  # Keep original object
            record.args = tuple(new_args)
        elif isinstance(record.msg, str) and len(record.msg) > self.max_length:
            # For f-strings or literals
            record.msg = record.msg[:self.max_length] + "..."
        return True
