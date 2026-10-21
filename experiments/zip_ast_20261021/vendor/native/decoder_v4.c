/* MathTokenizer native trie/DP: raw byte losslessness, optional tree spans.
 * Caller supplies immutable vocabulary/trie and fixed source-span masks.
 * No model cache or evaluation corpus dependency. All arrays caller-owned.
 */
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int legal(int a,int b,const int32_t *spans,int ns) {
    for(int k=0;k<ns;k++){
        int s=spans[2*k],e=spans[2*k+1];
        if(!(b<=s || a>=e || (a<=s && b>=e) || (a>=s && b<=e)))return 0;
    }
    return 1;
}

/* mode0 shortest; mode1 exact-v2 policy-MAP; mode2 additive learned token cost.
 * feature theta: length, logfreq, safe boundary, -atom split, -cross,
 * length/n, lookahead. minlen always includes all 256 byte fallback entries.
 */
int mt_decode(const uint8_t *data,int n,const int32_t *children,const int32_t *terminal,
              int maxlen,const int32_t *spans,int ns,int hard,
              const uint8_t *safe,const uint8_t *interior,const uint8_t *allowed,const double *logfreq,
              const double *theta,int ntheta,int mode,double splitfee,double crossfee,
              double lengthcost,const double *token_cost,int32_t *output) {
    if(n==0)return 0;
    /* Precompute interval compatibility O(n*maxlen + number_of_spans).
       Crossing iff an internal node start extends beyond token end, or an
       internal node end starts before token start. Equal boundaries allowed. */
    uint8_t *crossmask=(uint8_t*)calloc((size_t)n*(maxlen+1),1);
    int *maxend=(int*)calloc(n+1,sizeof(int));
    int *minstart=(int*)malloc((n+1)*sizeof(int));
    if(!crossmask||!maxend||!minstart){free(crossmask);free(maxend);free(minstart);return -1;}
    for(int i=0;i<=n;i++)minstart[i]=n+1;
    for(int k=0;k<ns;k++){
        int a=spans[2*k],b=spans[2*k+1];
        if(a>=0 && b<=n){if(b>maxend[a])maxend[a]=b;if(a<minstart[b])minstart[b]=a;}
    }
    if(ns)for(int a=0;a<n;a++){
        int mx=0,mn=n+1;
        for(int len=1;len<=maxlen && a+len<=n;len++){
            int end=a+len;
            crossmask[(size_t)a*(maxlen+1)+len]=(mx>end || mn<a);
            if(maxend[end]>mx)mx=maxend[end];if(minstart[end]<mn)mn=minstart[end];
        }
    }
    free(maxend);free(minstart);
    int *longest=(int*)malloc((n+1)*sizeof(int));
    double *best=(double*)malloc((n+1)*sizeof(double));
    int *choiceend=(int*)malloc(n*sizeof(int));
    int *choiceid=(int*)malloc(n*sizeof(int));
    int *ends=(int*)malloc((maxlen+1)*sizeof(int));
    int *ids=(int*)malloc((maxlen+1)*sizeof(int));
    double *logits=(double*)malloc((maxlen+1)*sizeof(double));
    double *lnlength=(double*)malloc((maxlen+1)*sizeof(double));
    if(lnlength)for(int l=1;l<=maxlen;l++)lnlength[l]=log((double)l);
    if(lnlength)lnlength[0]=0.;
    if(!longest||!best||!choiceend||!choiceid||!ends||!ids||!logits||!lnlength){
        free(longest);free(best);free(choiceend);free(choiceid);free(ends);free(ids);free(logits);free(lnlength);free(crossmask);return -1;
    }
    longest[n]=1;
    for(int a=0;a<n;a++){
        int node=0,l=1;
        for(int b=a;b<n && b-a<maxlen;b++){
            node=children[(size_t)node*256+data[b]];if(node<0)break;
            if(terminal[node]>=0)l=b-a+1;
        }longest[a]=l;
    }
    best[n]=0.;
    for(int a=mode==4?0:n-1;a>=0 && a<n;a=mode==4?choiceend[a]:a-1){
        int node=0,k=0;
        if(!allowed[a]){best[a]=INFINITY;choiceend[a]=-1;continue;}
        for(int b=a;b<n && b-a<maxlen;b++){
            node=children[(size_t)node*256+data[b]];if(node<0)break;
            int id=terminal[node];if(id<0)continue;
            int end=b+1,len=end-a;
            int crossing=crossmask[(size_t)a*(maxlen+1)+len];
            if((hard && crossing) || !allowed[end])continue;
            ends[k]=end;ids[k]=id;
            if(mode==1 || mode==3 || mode==4){
                double features[7]={lnlength[len],logfreq[id]/8.,(double)safe[end],
                    -(double)interior[end],-(double)crossing,(double)len/n,lnlength[longest[end]]};
                double logit=0.;for(int j=0;j<ntheta;j++)logit+=theta[j]*features[j];logits[k]=logit;
            }else{
                logits[k]=1.+splitfee*interior[end]+crossfee*crossing+lengthcost*len+(token_cost?token_cost[id]:0.);
            }k++;
        }
        if(k==0){best[a]=INFINITY;choiceend[a]=-1;continue;}
        if(k==1){choiceend[a]=ends[0];choiceid[a]=ids[0];
            best[a]=mode==4?0.:((mode==1 || mode==3)?0.:logits[0])+best[ends[0]];continue;}
        double logz=0.;
        if(mode==1 || mode==3){double mx=logits[0];for(int j=1;j<k;j++)if(logits[j]>mx)mx=logits[j];
            double sum=0.;for(int j=0;j<k;j++)sum+=exp(logits[j]-mx);logz=mx+log(sum);}
        double minimum=INFINITY;int selected=0;
        for(int j=0;j<k;j++){
            double cost=(mode==1 || mode==3)?logz-logits[j]:logits[j];
            if(mode==3)cost+=splitfee*interior[ends[j]]+crossfee*crossmask[(size_t)a*(maxlen+1)+ends[j]-a];
            double value=mode==4?-logits[j]:cost+best[ends[j]];
            if(value<minimum){minimum=value;selected=j;}
        }
        best[a]=minimum;choiceend[a]=ends[selected];choiceid[a]=ids[selected];
    }
    int count=0,pos=0;
    while(pos<n){if(choiceend[pos]<0){count=-2;break;}output[count++]=choiceid[pos];pos=choiceend[pos];}
    free(longest);free(best);free(choiceend);free(choiceid);free(ends);free(ids);free(logits);free(lnlength);free(crossmask);return count;
}

/* AST-v4 exact shortest-path specialization. Compatibility accumulates during
 * actual trie traversal; no n*maxlen crossing table or policy-feature passes.
 * Ties keep the earliest matching end, as in mt_decode(mode=0). */
int mt_shortest(const uint8_t *data,int n,const int32_t *children,const int32_t *terminal,
                int maxlen,const int32_t *spans,int ns,int hard,
                const uint8_t *allowed,int32_t *output) {
    if(n==0)return 0;
    int *maxend=(int*)calloc((size_t)n+1,sizeof(int));
    int *minstart=(int*)malloc(((size_t)n+1)*sizeof(int));
    int *best=(int*)malloc(((size_t)n+1)*sizeof(int));
    int *choiceend=(int*)malloc((size_t)n*sizeof(int));
    int *choiceid=(int*)malloc((size_t)n*sizeof(int));
    if(!maxend||!minstart||!best||!choiceend||!choiceid){
        free(maxend);free(minstart);free(best);free(choiceend);free(choiceid);return -1;
    }
    for(int i=0;i<=n;i++)minstart[i]=n+1;
    if(hard)for(int k=0;k<ns;k++){
        int a=spans[2*k],b=spans[2*k+1];
        if(a>=0 && b<=n && a<b){
            if(b>maxend[a])maxend[a]=b;if(a<minstart[b])minstart[b]=a;
        }
    }
    best[n]=0;
    for(int a=n-1;a>=0;a--){
        best[a]=n+1;choiceend[a]=-1;
        if(!allowed[a])continue;
        int node=0,mx=0,mn=n+1;
        for(int b=a;b<n && b-a<maxlen;b++){
            node=children[(size_t)node*256+data[b]];if(node<0)break;
            int end=b+1,id=terminal[node];
            int crossing=mx>end || mn<a;
            if(id>=0 && allowed[end] && !crossing && best[end]<n+1){
                int value=1+best[end];
                if(value<best[a]){best[a]=value;choiceend[a]=end;choiceid[a]=id;}
            }
            if(maxend[end]>mx)mx=maxend[end];if(minstart[end]<mn)mn=minstart[end];
        }
    }
    int count=0,pos=0;
    while(pos<n){
        if(choiceend[pos]<0){count=-2;break;}
        output[count++]=choiceid[pos];pos=choiceend[pos];
    }
    free(maxend);free(minstart);free(best);free(choiceend);free(choiceid);return count;
}

/* Single traversal for all matching substring token occurrences. */
int mt_matches(const uint8_t *data,int n,const int32_t *children,const int32_t *terminal,
               int maxlen,int32_t *starts,int32_t *ends,int32_t *ids,int cap){
    int count=0;
    for(int a=0;a<n;a++){
        int node=0;
        for(int b=a;b<n && b-a<maxlen;b++){
            node=children[(size_t)node*256+data[b]];if(node<0)break;
            int id=terminal[node];if(id<0)continue;
            if(count>=cap)return -1;
            starts[count]=a;ends[count]=b+1;ids[count]=id;count++;
        }
    }return count;
}
