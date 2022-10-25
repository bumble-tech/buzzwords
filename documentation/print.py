from npdoc_to_md import render_md_from_obj_docstring
from pkg_resources import get_distribution

from buzzwords import Buzzwords


def convert_docs(object, name):
    """
    Print numpy doc from a Python object converted to markdown format

    Arguments
    ---------
    object : object
        Object to print docstring for
    name : str
        name of object

    Returns
    -------
    formatted_string : str
        Docstring converted to markdown, with horizontal rule and newlines appended
    """

    # Return either the docstring or an empty string if it doesn't have one
    md = render_md_from_obj_docstring(object, name) or ''

    # Add newlines and horizontal rule
    formatted_string = f'{md}\n\n***\n\n'

    return formatted_string


def main():
    """
    Print out the Buzzwords documentation from the docstrings in markdown format
    """

    # Print version number and Buzzwords docs
    print("[<< Back](..)\n")
    print(f'# v{get_distribution("buzzwords").version}\n\n')
    print(convert_docs(Buzzwords, 'Buzzwords'))

    # Iterate through all functions without '__' and print docstrings for each
    for function in dir(Buzzwords):
        if not function.startswith('__'):
            print(
                convert_docs(
                    getattr(Buzzwords, function),
                    function
                )
            )


if __name__ == '__main__':
    main()
