r"""
This is an overview of the PEP 8 formatting guidelines. Refer to the user full
guide for further details, which can be found at https://github.com/python/peps/blob/main/pep-0008.txt.

Imports
=======
Imports are always put at the top of the file.

Imports should be on separate lines.

Examples
--------

Correct:

>>> import os
>>> import sys

Wrong:

>>> import sys, os

Imports should be grouped in the following order:

#. Standard library imports.
#. Related third party imports.
#. Local application/library specific imports.

Naming convention
=================

Naming convention is as follows:

- Functions: All lowercase, with words separated by underscores as necessary to improve readability
- Variables: Same convention as function names
- Classes: Start each word with a capital letter. Do not separate words with underscores. This style is called camel case
- Methods: Use a lowercase word or words. Separate words with underscores to improve readability.
- Constants: Use an uppercase single letter, word, or words. Separate words with underscores to improve readability.
- Modules: Use a short, lowercase word or words. Separate words with underscores to improve readability.
- Packages: Use a short, lowercase word or words. Do not separate words with underscores.

Never use l, o, or I single letter names as these can be mistaken for 1 and 0, depending on typeface
"""


#EXAMPLES:
#Package and module
from datetime import date

#Constant


#Functions and variables
def my_function(x,my_variable):
    # an example of documentation for this function
    """
    Print out ``Hello World`` and return nothing

    :param x: Explanation of this input parameter
    :param my_variable: Explanation of this input parameter
    :type x: int
    :type my_variable: str
    :return: This function returns nothing
    :rtype: None
    """
    print('Hello world')

#Classes and method
def MyClass(x,y):
    print('Hello world')

    #First method example
    def method(x):
        print('Hello again')

    #Second method example
    def class_method(x):
        print('Hello again')

######################################################################
# Identation
# ==========
# -Lines should be limited to 79 characters, indicated by this vertical line--->
# -Use 4 spaces instead of tabs (you can modify settings so that tab key produces 4 spaces instead of tab character)
# -Continuation lines should align wrapped elements either vertically using Python’s implicit line joining inside
# parentheses, brackets and braces, or using a hanging indent.
# -When using a hanging indent the following should be considered; there should be no arguments on the first line
# and further indentation should be used to clearly distinguish itself as a continuation line
# -The 4-space rule is optional for continuation lines


# CORRECT EXAMPLES:
# Add 4 spaces (an extra level of indentation) to distinguish arguments from the rest.
def long_function_name(
        var_one, var_two, var_three,
        var_four):
    # an example of documentation for this function
    """
    Print out the first argument and return nothing

    :param var_one: Explanation of this input parameter
    :type var_one: str
    :return: This function returns nothing
    :rtype: None
    """
    print(var_one)


# Aligned with opening delimiter.
#foo = long_function_name(var_one, var_two,
#                         var_three, var_four)


# Hanging indents should add a level.
#foo = long_function_name(
#    var_one, var_two,
#    var_three, var_four)

# Hanging indents *may* be indented to other than 4 spaces.
#foo = long_function_name(
#  var_one, var_two,
#  var_three, var_four)

# INCORRECT EXAMPLES:
# Arguments on first line forbidden when not using vertical alignment.
#foo = long_function_name(var_one, var_two,
#    var_three, var_four)

# Further indentation required as indentation is not distinguishable.
def long_function_name2(
    var_one, var_two, var_three,
    var_four):
    print(var_one)

#%% Line breaks, blank lines and blank spaces
#-Line break should be before binary operators. Easier to match operands.
#-Surround top-level function and class definitions with two blank lines.
#-Method definitions inside a class are surrounded by a single blank line.
#-Extra blank lines may be used (sparingly) to separate groups of related functions.
#Blank lines may be omitted between a bunch of related one-liners (e.g. a set of dummy implementations).
#-Use blank lines in functions, sparingly, to indicate logical sections.

#EXAMPLE:
#Correct example
#income = (gross_wages
#          + taxable_interest
#          + (dividends - qualified_dividends)
#          - ira_deduction
#          - student_loan_interest)

#Wrong example
#income = (gross_wages +
#          taxable_interest +
#          (dividends - qualified_dividends) -
#          ira_deduction -
#          student_loan_interest)

#Avoid extraneous whitespace in the following situations:

# Correct:
#spam(ham[1], {eggs: 2})              #inside parentheses, brackets or braces
#foo = (0,)                           #Between a trailing comma and a following close parenthesis
#if x == 4: print(x, y); x, y = y, x  #before a comma, semicolon, or colon

# Wrong:
#spam( ham[ 1 ], { eggs: 2 } )
#bar = (0, )
#if x == 4 : print(x , y) ; x , y = y , x

#%% Comments
#-Comments that contradict the code are worse than no comments. Always make a
# priority of keeping the comments up-to-date when the code changes!
#-Comments should be complete sentences. The first word should be capitalized,
# unless it is an identifier that begins with a lower case letter (never alter the case of identifiers!).

#Block comments:
#-Block comments generally consist of one or more paragraphs built out of
# complete sentences, with each sentence ending in a period.
#-You should use two spaces after a sentence-ending period in multi-sentence
# comments, except after the final sentence.

#Inline comments:
#-Use inline comments sparingly. An inline comment is a comment on the same line as a statement.
#-Inline comments should be separated by at least two spaces from the statement.
#-They should start with a # and a single space
