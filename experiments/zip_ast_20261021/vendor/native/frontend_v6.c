/* Lossless LaTeX byte lexer and source binding tree. No TeX execution.
 * Lexical constraints apply to corpus-derived atoms; unknown units use bytes.
 * This is a concrete syntax tree, not a universal mathematical semantic AST.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int letter(uint8_t c){return (c>='A'&&c<='Z')||(c>='a'&&c<='z');}
static int white(uint8_t c){return c==' '||c=='\t'||c=='\r'||c=='\n';}
static int unit_end(const uint8_t*d,int n,int a){
    int b=a+1;
    if(d[a]=='\\'&&b<n){
        if(letter(d[b]))while(b<n&&letter(d[b]))b++;
        else {int len=(d[b]<128?1:d[b]<224?2:d[b]<240?3:4);b+=len;}
    }else if(d[a]>=128){int len=d[a]<224?2:d[a]<240?3:4;b=a+len;}
    else if(d[a]>='0'&&d[a]<='9'){
        while(b<n&&d[b]>='0'&&d[b]<='9')b++;
        if(b+1<n&&d[b]=='.'&&d[b+1]>='0'&&d[b+1]<='9'){
            b++;while(b<n&&d[b]>='0'&&d[b]<='9')b++;
        }
    }
    return b<n?b:n;
}

static int lookup(const uint8_t*d,int a,int b,const int32_t*ch,const int32_t*terminal){
    int node=0;
    for(int i=a;i<b;i++){node=ch[(size_t)node*256+d[i]];if(node<0)return -1;}
    return terminal[node];
}

/* Supported algebraic subset: arithmetic precedence, implicit products,
 * groups, scripts, fractions and roots. Other valid LaTeX keeps the binding
 * CST. Unknown syntax never fabricates a mathematical semantic AST. */
typedef struct {const uint8_t*d;int n,p,depth,ok,k;int32_t*out;} ExprParser;
typedef struct {int a,b;} ExprSpan;
static void spaces(ExprParser*p){while(p->p<p->n&&white(p->d[p->p]))p->p++;}
static void node(ExprParser*p,ExprSpan s){if(s.b>s.a){p->out[2*p->k]=s.a;p->out[2*p->k+1]=s.b;p->k++;}}
static ExprSpan expression(ExprParser*p,int minimum);
static ExprSpan atom(ExprParser*p,int scripts);
static ExprSpan primary(ExprParser*p){return atom(p,1);}
static int command_is(ExprParser*p,int a,int b,const char*name){return b-a==(int)strlen(name)&&memcmp(p->d+a,name,(size_t)(b-a))==0;}
static ExprSpan atom(ExprParser*p,int scripts){
    spaces(p);int a=p->p;ExprSpan result={a,a};
    if(a>=p->n||++p->depth>128){p->ok=0;p->depth--;return result;}
    uint8_t c=p->d[a];
    if(c=='+'||c=='-'){
        p->p++;ExprSpan child=expression(p,4);result.b=child.b;
    }else if(c=='{'||c=='('){
        p->p++;expression(p,0);spaces(p);
        if(p->p>=p->n||p->d[p->p]!=(c=='{'?'}':')'))p->ok=0;
        else p->p++;
        result.b=p->p;
    }else if(letter(c)||c>=128||(c>='0'&&c<='9')){
        p->p=unit_end(p->d,p->n,a);result.b=p->p;
    }else if(c=='\\'){
        int end=unit_end(p->d,p->n,a);p->p=end;
        int arity=(command_is(p,a,end,"\\frac")||command_is(p,a,end,"\\dfrac")||command_is(p,a,end,"\\tfrac"))?2:command_is(p,a,end,"\\sqrt")?1:0;
        if(!arity)p->ok=0;
        for(int i=0;i<arity&&p->ok;i++){
            spaces(p);if(p->p>=p->n||p->d[p->p]!='{'){p->ok=0;break;}
            atom(p,0);
        }result.b=p->p;
    }else p->ok=0;
    if(p->ok)node(p,result);
    while(p->ok&&scripts){
        spaces(p);if(p->p>=p->n||(p->d[p->p]!='^'&&p->d[p->p]!='_'))break;
        p->p++;ExprSpan argument=atom(p,0);result.b=argument.b;node(p,result);
    }
    p->depth--;return result;
}
static ExprSpan expression(ExprParser*p,int minimum){
    ExprSpan left=primary(p);
    while(p->ok){
        spaces(p);int a=p->p;if(a>=p->n)break;
        uint8_t c=p->d[a];int precedence=0,end=a+1,implicit=0;
        if(c=='='||c=='<'||c=='>')precedence=1;
        else if(c=='+'||c=='-')precedence=2;
        else if(c=='*'||c=='/')precedence=3;
        else if(letter(c)||c>=128||(c>='0'&&c<='9')||c=='{'||c=='('){precedence=3;implicit=1;}
        else if(c=='\\'){
            end=unit_end(p->d,p->n,a);
            if(command_is(p,a,end,"\\cdot")||command_is(p,a,end,"\\times")||command_is(p,a,end,"\\div"))precedence=3;
            else if(command_is(p,a,end,"\\leq")||command_is(p,a,end,"\\geq")||command_is(p,a,end,"\\neq"))precedence=1;
            else {precedence=3;implicit=1;}
        }
        if(!precedence||precedence<minimum)break;
        if(!implicit)p->p=end;
        ExprSpan right=expression(p,precedence+1);left.b=right.b;if(p->ok)node(p,left);
    }return left;
}

/* out has n*2 entries: an exact byte-coverage lexical partition. */
int mt_lex(const uint8_t*d,int n,int32_t*out){
    int k=0;
    for(int a=0;a<n;){int b=unit_end(d,n,a);out[2*k]=a;out[2*k+1]=b;k++;a=b;}
    return k;
}

/* Return >=0 source spans; caller reserves 8*n pairs. Group mismatch never
 * prevents byte encoding, and the return carries no semantic validity claim.
 */
int mt_tree(const uint8_t*d,int n,int32_t*out){
    int *close=malloc(((size_t)n+1)*sizeof(int));
    int *stack=malloc(((size_t)n+1)*sizeof(int));
    int *completed=malloc(((size_t)n+1)*sizeof(int));
    if(!close||!stack||!completed){free(close);free(stack);free(completed);return -1;}
    for(int i=0;i<=n;i++){close[i]=-1;completed[i]=-1;}
    int depth=0,k=0;
    for(int a=0;a<n;){
        int b=unit_end(d,n,a);
        out[2*k]=a;out[2*k+1]=b;k++;
        if(d[a]=='{'&&b==a+1)stack[depth++]=a;
        else if(d[a]=='}'&&b==a+1&&depth){int start=stack[--depth];close[start]=b;completed[b]=start;out[2*k]=start;out[2*k+1]=b;k++;}
        a=b;
    }
    int previous=0;
    for(int a=0;a<n;){
        int b=unit_end(d,n,a);
        if(d[a]=='\\'&&a+1<n&&letter(d[a+1])){
            int p=b;
            while(p<n&&white(d[p]))p++;
            // Generic consecutive braced argument binding, no command list.
            while(p<n&&d[p]=='{'&&close[p]>p){
                p=close[p];out[2*k]=a;out[2*k+1]=p;k++;completed[p]=a;
                while(p<n&&white(d[p]))p++;
            }
        }
        if((d[a]=='^'||d[a]=='_')&&a>0){
            int p=b;while(p<n&&white(d[p]))p++;
            if(p<n){int e=d[p]=='{'&&close[p]>p?close[p]:unit_end(d,n,p);
                out[2*k]=previous;out[2*k+1]=e;k++;
                completed[e]=previous;
            }
        }
        if(!white(d[a]))previous=completed[b]>=0?completed[b]:a;
        a=b;
    }
    if(n){out[2*k]=0;out[2*k+1]=n;k++;}
    ExprParser parser={d,n,0,0,1,k,out};expression(&parser,0);spaces(&parser);
    if(parser.ok&&parser.p==n)k=parser.k;
    free(close);free(stack);free(completed);return k;
}

/* Known complete commands and UTF-8 codepoints are protected. */
int mt_allowed(const uint8_t*d,int n,const int32_t*ch,const int32_t*terminal,uint8_t*out){
    memset(out,1,(size_t)n+1);int missing=0;
    for(int a=0;a<n;){int b=unit_end(d,n,a);
        if(b-a>1&&(d[a]=='\\'||d[a]>=128)){
            if(lookup(d,a,b,ch,terminal)>=0)memset(out+a+1,0,(size_t)(b-a-1));else missing++;
        }a=b;
    }return missing;
}

/* Map lexical atoms to a UTF-8 PUA alphabet for unmodified Rust BPE.
 * Unknown units escape one byte at a time. Every byte has one base symbol.
 */
int mt_symbols(const uint8_t*d,int n,const int32_t*ch,const int32_t*terminal,uint8_t*out){
    int k=0;
    for(int a=0;a<n;){int b=unit_end(d,n,a);int id=lookup(d,a,b,ch,terminal);
        if(id>=0){int cp=0xE000+id;out[k++]=0xE0|(cp>>12);out[k++]=0x80|((cp>>6)&63);out[k++]=0x80|(cp&63);a=b;}
        else for(;a<b;a++){int cp=0xE000+d[a];out[k++]=0xE0|(cp>>12);out[k++]=0x80|((cp>>6)&63);out[k++]=0x80|(cp&63);}
    }return k;
}
