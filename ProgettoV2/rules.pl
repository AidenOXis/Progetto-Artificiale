% Regole Prolog migliorate

% Se un nodo interagisce ma non lo dichiara, mente
mente(X) :- interazione(X,Y), \+ dichiarazione(X,Y).

% Un bugiardo ha mentito più volte
bugiardo(X) :- mente(X), mente(Z), X \= Z.

% Cerca di incastrare un innocente
depista(X, Y) :- dichiarazione(X,Y), \+ interazione(X,Y), innocente(Y).

% Cambia dichiarazione nel tempo
alibi_falso(X) :- dichiarazione(X,Y), dichiarazione_precedente(X,Z), Y \= Z.

% innocente = non ha interagito con vittime
innocente(X) :- \+ (interazione(X,Y), vittima(Y)).

% sospetto base: interazione con 2 vittime
colpevole(X) :- 
    interazione(X,Y), vittima(Y),
    interazione(X,Z), vittima(Z),
    X \= Y, X \= Z.


