import sys

def error_message_detail(error, sys_module: sys):
    _, _, exc_tb = sys_module.exc_info()

    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"at line number [{line_number}] "
        f"with message [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message,
            sys_module=error_detail
        )

    def __str__(self):
        return self.error_message