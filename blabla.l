%option noyywrap
%{
#include <stdio.h>
#include "FichiersH/commun.h"
#include "miniC.tab.h"
#include <stdlib.h>

%}

D                       [0-9]
L                       [a-zA-Z_]

%%


"break"                 { return BREAK; }
"case"                  { return CASE; }
"const"                 { return CONSTANTE; }
"default"               { return DEFAULT; }
"else"                  { return ELSE; }
"extern"                { return EXTERN; }
"for"                   { return FOR; }
"if"                    { return IF; }
"int"                   { return INT; }
"return"                { return RETURN; }
"switch"                { return SWITCH; }
"void"                  { return VOID; }
"while"                 { return WHILE; }


{L}({L}|{D})*           { yylval.val.chaine = strdup(yytext); return IDENTIFICATEUR ; }  /* Identificateurs, strdup sert a alloué de la memoire, alterenative a malloc  */


{D}+                    {  yylval.val.chiffre = atoi(yytext); return CONSTANTE; }

"+"                     { return PLUS; }
"-"                     { return MOINS; }
"*"                     { return MUL; }
"/"                     { return DIV; }
"%"                     { return '%'; }
"&&"                    { return LAND; }
"||"                    { return LOR; }
"!"                     { return NOT; }
"=="                    { return EQ; }
"!="                    { return NEQ; }
"<"                     { return LT; }
">"                     { return GT; }
"<="                    { return LEQ; }
">="                    { return GEQ; }
"<<"                    { return LSHIFT; }
">>"                    { return RSHIFT; }


\/\*([^*]|\*+[^*/])*\*+\/    { ; } /* Commentaire ignoré */ 

[\n]           { yylineno++; }
[ \r\t\f\v]+   { ; } /* Espaces blancs */


.                       {  return yytext[0] ; }  

%%