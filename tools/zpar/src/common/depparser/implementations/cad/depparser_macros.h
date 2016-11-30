// Copyright (C) University of Oxford 2010
#ifndef _GENERAL_DEPPARSER_MACROS_H
#define _GENERAL_DEPPARSER_MACROS_H

#define SIMPLE_HASH

// early update? 
#define EARLY_UPDATE

//use hamming loss
#define USE_HAMLOSS

// maxprec
//#define USE_MAXPREC

// local training? it will affect the setting of early update and agenda size
// if you want to experiment with local training, define this when you train
// it will automatically set beam1
// but undefine this when you decode with beam more than one
// using the model you trained with this defined
//#define LOCAL_LEARNING

// The size of agenda
#define AGENDA_SIZE 64

//label
#ifdef LABELED
const unsigned DEP_LABEL_COUNT=CDependencyLabel::MAX_COUNT;
#else
const unsigned DEP_LABEL_COUNT = 1;
#endif

typedef double SCORE_TYPE ;
#include "action.h"

// supertag
#define SR_SUPERTAG 1

// force the tree to be single-rooted or allow multiple pseudo roots
//#define FRAGMENTED_TREE

// the implementation supports the extraction of features as a command
#define SUPPORT_FEATURE_EXTRACTION

// The size of a sentence and the words
const unsigned MAX_SENTENCE_SIZE = 256 ; 
const unsigned MAX_SENTENCE_SIZE_BITS = 8 ; 

// The round of start MAXPREC
const unsigned MIN_START_ROUND_MAXPREC = 39832*5;
// The probability of using early-update
const float EARLY_UP_DATE_PROB = 0.9;

// normalise link size and the direction
inline int encodeLinkDistance(const int &head_index, const int &dep_index) {
   static int diff;
   diff = head_index - dep_index;
   assert(diff != 0); 
   if (diff<0)
      diff=-diff;
   if (diff>10) diff = 6; 
   else if (diff>5) diff = 5; 
   return diff;
}

// arity direction
enum ARITY_DIRECTION { ARITY_DIRECTION_LEFT=0, ARITY_DIRECTION_RIGHT=1 } ;

#ifdef LOCAL_LEARNING
#define EARLY_UPDATE
#define AGENDA_SIZE 1
#endif

#endif
