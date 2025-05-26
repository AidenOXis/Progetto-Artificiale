% If a node interacts but doesn’t declare it, they’re lying
lies(X) :- interaction(X,Y), \+ declaration(X,Y).