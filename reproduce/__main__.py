"""``python -m reproduce`` - the article's tables, macros and figures, from the released tables."""
from . import figures, macros, tables

tables.main()
macros.main([])
figures.main()
