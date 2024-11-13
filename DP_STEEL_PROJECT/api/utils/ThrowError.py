import sys
import traceback

def ThrowError():
    '''Throws an error with traceback.'''
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)   