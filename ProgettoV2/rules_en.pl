% Lying: interaction exists but not declared
lies(X) :- interaction(X,Y), \+ declaration(X,Y).

% If a node declared someone they didn't interact with → fake declaration
fake_declaration(X) :- declaration(X,Y), \+ interaction(X,Y).

% If a node interacts with more than one victim → possibly a killer
multi_victim_contact(X) :- victim(V1), victim(V2), V1 \= V2, interaction(X, V1), interaction(X, V2).

% If a node planted evidence AND made fake declarations → very suspicious
suspicious_behavior(X) :- planted(X), fake_declaration(X).

% If a node is silent but highly connected → manipulator
silent_operator(X) :- findall(Y, interaction(X,Y), L1), length(L1, N1),
                      findall(Y, declaration(X,Y), L2), length(L2, N2),
                      N1 > 3, N2 < 2.
