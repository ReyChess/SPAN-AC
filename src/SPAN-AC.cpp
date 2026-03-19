/*	@ este programa construye un clasificador basado en CAR's para lo cual realiza los distintos pasos:
 * 	@ a.- Lectura del dataset
 * 	@ b.- Formacion de 10 sub-conjuntos disjuntos del dataset de forma tal que en cada sub-conjunto
 * 				esten presentes ejemplos de todas las clases.
 * 	@ c.- A partir de los sub-conjuntos realizar las fases de entrenamiento o construccion del clasificador
 * 				utilizando 9 de los 10 sub-conjuntos y a continuacion evaluacion del clasificador construido utilizando
 * 				el sub-conjunto restante. Este paso se descompone en:
 * 					i.- Formacion del file del dataset a partir de los sub-conjuntos a utilizar en el entrenamiento.
 * 					ii.- Ejecucion del algoritmo de extraccion de CAR's a partir del file del dataset creado en (i).
 * 					iii.- Construccion del clasificador con las reglas obtenidas en el paso anterior.
 * 					iv.- Evaluacion del clasificador con el conjunto faltante del entrenamiento.
 * 					v.- Reporte de la eficiencia alcanzada. 
 * 	@ d.- Ejecutar el paso c con cada una de las combinaciones de 9 que se pueden extraer de 10 sub-conjuntos.	
 * 	@ e. Calculo de la eficiencia promedio del proceso.
 *  NOTA :- Aqui la clase por defecto se selecciona en dependencia de la clase cuyo conjunto de reglas representativo
 * 					tenga un promedio de netconf mayor
 * */

// directivas incluidas

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <stdint.h>
#include <inttypes.h>


// ------------------ Partition / fold file signature helpers ------------------
// When doing repeated runs with externally-generated folds (DivideEn10), we want
// to record *which* partition was used, so later stability analysis can be done
// correctly across runs.
// We compute a small signature per fold file based on (size, mtime).
static inline void file_signature(const char* path, long long* out_size, long long* out_mtime)
{
  struct stat st;
  if (stat(path, &st) == 0) {
    *out_size = (long long)st.st_size;
    *out_mtime = (long long)st.st_mtime;
  } else {
    *out_size = -1;
    *out_mtime = -1;
  }
}

// ================== Tipos base (DEBEN estar completos antes de usarlos) ==================
// En C/C++ NO se puede acceder a miembros de un struct declarado solo con "forward declaration".
// El error que viste ("invalid use of incomplete type") ocurre exactamente por eso.
// Por tanto, definimos aquí los structs tal y como están en tu main.cpp original.

typedef struct
{
  int * elementos, longitud, tamanno;
} CConjunto;

typedef struct
{
  CConjunto * antecedente;
  int clase; // etiqueta de clase (siempre positiva)
  unsigned char negated; // 0: A->c (positiva), 1: A->neg(c) (negativa)
  double netconfREGLA, interesDONG, soporteREGLA, soporteANTECEDENTE, soporteCLASE;
} CCAR;

typedef struct
{
  CConjunto * items;
  int clase;
} CTransaccion;

// ================== Stable rule selection (deterministic & smoothed) ==================
// NOTA:
//  - El esquema anterior ("K dinámico" con K = N_min) puede ser muy inestable cuando
//    N_min es pequeño (pocas reglas por clase cubriendo la instancia). En esos casos,
//    el promedio de netconf se vuelve muy ruidoso y el ganador puede cambiar drásticamente
//    de un fold a otro.
//
// Este bloque implementa una selección más estable:
//  1) Usa un K fijo acotado (K_MAX) por clase.
//  2) Promedio ponderado por rango (reglas mejores pesan más).
//  3) Suavizado Bayesiano/shrinkage hacia un prior global (MU0) con fuerza BETA.
//  4) Desempates deterministas: score, luego prior, luego id de clase.
//
// Ajusta estos parámetros si lo deseas (valores conservadores):
#ifndef STABLE_K_MAX
#define STABLE_K_MAX 12
#endif

#ifndef STABLE_BETA
#define STABLE_BETA 2.0
#endif

// Negatives should be more conservative than positives.
// We regularize negative evidence harder to avoid one spurious negative dominating.
#ifndef STABLE_BETA_NEG
#define STABLE_BETA_NEG 15.0
#endif

#ifndef STABLE_LAMBDA_NEG
#define STABLE_LAMBDA_NEG 0.25
#endif

// Adaptive negative weight: negatives matter most when positives are extremely close.
// tau controls how fast the negative influence decays as the positive gap grows.
#ifndef STABLE_TAU
#define STABLE_TAU 0.01
#endif

// Usar CARs negativas SOLO para romper empates en la evidencia positiva.
// Un "empate" se define como clases con ScorePos a distancia <= STABLE_TIE_EPS del máximo.
#ifndef STABLE_TIE_EPS
// Near-tie threshold (POS evidence units). For FLARE, 0.002 is a safe starting point.
#define STABLE_TIE_EPS 1e-6
#endif

// Numerical epsilon (do NOT tune this for behavior).
#ifndef STABLE_NUM_EPS
#define STABLE_NUM_EPS 1e-12
#endif

// Para evitar que la evidencia negativa domine (especialmente en datasets ruidosos),
// limitamos el número de negativas consideradas por clase.
#ifndef STABLE_K_NEG_MAX
#define STABLE_K_NEG_MAX 2
#endif

#ifndef STABLE_LAMBDA_PRIOR
// Small prior term to avoid over-predicting minority classes in highly imbalanced datasets.
// This term is used in the BASE score (always) and in near-tie resolution.
#define STABLE_LAMBDA_PRIOR 0.15
#endif

#ifndef STABLE_POS_MIN
// If the best positive evidence is below this threshold, treat the instance as weakly covered
// and fall back to the default class.
#define STABLE_POS_MIN -1e300
#endif

static inline double stable_rank_weight(int rank0)
{
  // rank0 = 0 para la mejor regla, 1 para la segunda, ...
  // pesos decrecientes: 1, 1/2, 1/3, ...
  return 1.0 / (double)(rank0 + 1);
}

// Evidencia POSITIVA (A->c): promedio ponderado top-K + shrinkage hacia mu0_pos
static inline double stable_score_pos_class(
    int c,
    CConjunto **reglasCUBREN_POS,
    CCAR ***listaCCAR,
    double mu0_pos)
{
  int Nc = reglasCUBREN_POS[c]->longitud;
  if (Nc <= 0) return -1e300;

  int K = (Nc < STABLE_K_MAX ? Nc : STABLE_K_MAX);

  double num = 0.0, den = 0.0;
  for (int j = 0; j < K; ++j) {
    int idx = reglasCUBREN_POS[c]->elementos[j];
    double w = stable_rank_weight(j);
    num += w * listaCCAR[c][idx]->netconfREGLA; // netconf positivo (A->c)
    den += w;
  }

  double avg = (den > 0.0 ? (num / den) : 0.0);

  // shrinkage hacia mu0_pos (baseline de netconf de reglas positivas que cubren la instancia)
  double shrunk = (avg * (double)K + STABLE_BETA * mu0_pos) / ((double)K + STABLE_BETA);

  return shrunk;
}

// Evidencia NEGATIVA (A->neg(c)): penalización (netconf positivo, semántica "contra c")
// La encogemos hacia 0 (sin penalización) para no castigar clases sin reglas negativas.
static inline double stable_score_neg_class(
    int c,
    CConjunto **reglasCUBREN_NEG,
    CCAR ***listaCCAR)
{
  int Nc = reglasCUBREN_NEG[c]->longitud;
  if (Nc <= 0) return 0.0;

  // Penalize only the strongest negative evidence (max), not a sum.
  // This avoids over-penalization and matches the semantic question:
  // "Is there a strong reason to reject this class?"
  //
  // We still use a small K cap to reduce the chance of one noisy rule
  // being the only considered evidence, and we shrink toward 0.
  int K = (Nc < STABLE_K_NEG_MAX ? Nc : STABLE_K_NEG_MAX);
  double maxNeg = 0.0;
  for (int j = 0; j < K; ++j) {
    int idx = reglasCUBREN_NEG[c]->elementos[j];
    double v = listaCCAR[c][idx]->netconfREGLA; // netconf positivo (A->neg(c))
    if (v > maxNeg) maxNeg = v;
  }

  // Shrink toward 0 with a stronger beta for negatives.
  // Scale by K so that multiple independent negatives increase reliability.
  double shrunk = (maxNeg * (double)K) / ((double)K + STABLE_BETA_NEG);
  return shrunk;
}



// definicion de variables
int * clasesCOLECCION, cCLASES = 0 , tCLASES = 100, tBuffer = 2000, defaultCLASS = 0;
const int tamanno_arregloCjt = 6000;
char ** nombres_files;
char ** nombres_REGLAS;

double eficienciaPROMEDIO = 0, eficienciaEXPERIMENTO = 0;

CCAR *** listaCCAR;
int * cantidadCCAR, * totalCCAR;
double ** aveREGLA, ** desvREGLA;
int * itemsEJEMPLO;
CConjunto ** reglasCUBREN_POS, ** reglasCUBREN_NEG, ** reglasCUBREN_POS_CANTIDAD;
int claseASIGNADA;
double * averageWEIGHT;

// ================== Stability logging (CSV) ==================
// This file is intentionally lightweight: it enables direct measurement of
// decision stability (agreement/entropy) and tie-region behavior across
// repeated CV runs. It does NOT require access to baseline models.
static int STABILITY_RUN_ID = 0; // optional: passed as argv[2]

static inline long file_size_bytes(const char* path)
{
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return (long)st.st_size;
}

static inline void csv_write_header_if_empty(FILE* f, const char* path, const char* header)
{
  if (!f) return;
  if (file_size_bytes(path) == 0) {
    fprintf(f, "%s\n", header);
    fflush(f);
  }
}
double * priorClase;   // prior probability for each class (same index as clasesCOLECCION)

// definicion de variables

// prototipos de los metodos
int process_arguments(int argc, char *argv[]);
void Leer_File_CLASSES();
void Leer_Clasificador(char * file_clasificador);
void Construye_ClasificadorEXACTO(char * file_clasificador,char * file_prueba, int fold_idx);
void Construye_ClasificadorFUZZY(char * file_clasificador,char * file_prueba, double b0);
void Clasifica_Dataset(int argc, char *argv[]);

// prototipos de los metodos

// implementacion de los metodos
int process_arguments(int argc, char *argv[])
{
  if (argc != 2 && argc != 3)
  {
    printf("\nError! Usage: %s <dataset_name> [run_id]\n", argv[0]);
    return 1;  }

  if (argc == 3) {
    STABILITY_RUN_ID = atoi(argv[2]);
  }
  
  return 0;
}

void Leer_File_CLASSES()
{
  clasesCOLECCION = 0;
  clasesCOLECCION = (int*)malloc(sizeof(int)*tCLASES);
  priorClase = 0;
  priorClase = (double*)malloc(sizeof(double)*tCLASES);
  cCLASES = 0;

  FILE * ficheroREAD = 0;
  ficheroREAD = fopen("Classes.dat","r");

  // buffer que almacenara lo leido
  char * buffer = 0;
  buffer = (char *)malloc(sizeof(char)*tBuffer);
  char * eof = 0;
  char * temp = 0;
  temp = (char *)malloc(sizeof(char)*20);
  int inicio = 0, j = 0;

  double totalCount = 0.0;   // suma de los conteos de todas las clases

  eof = fgets(buffer,tBuffer,ficheroREAD);
  while (eof != 0)
  {
    // ----- leer el label de la clase (primer numero) -----
    inicio = 0; j = 0;
    while (buffer[inicio]!= ' ' && buffer[inicio] != '\t' && buffer[inicio] != '\n' && buffer[inicio] != '\0')
    {
      temp[j] = buffer[inicio];
      inicio++; j++;
    }
    temp[j] = '\0';

    // si es necesario, agrandar arreglos
    if (cCLASES == tCLASES)
    {
      tCLASES *= 2;
      clasesCOLECCION = (int*)realloc(clasesCOLECCION, sizeof(int)*tCLASES);
      priorClase      = (double*)realloc(priorClase,      sizeof(double)*tCLASES);
    }

    clasesCOLECCION[cCLASES] = atoi(temp);

    // ----- leer el conteo de la clase (segundo numero) -----
    // saltar espacios
    while (buffer[inicio] == ' ' || buffer[inicio] == '\t')
      inicio++;

    j = 0;
    while (buffer[inicio]!= ' ' && buffer[inicio] != '\t' && buffer[inicio] != '\n' && buffer[inicio] != '\0')
    {
      temp[j] = buffer[inicio];
      inicio++; j++;
    }
    temp[j] = '\0';

    int count = atoi(temp);
    priorClase[cCLASES] = (double)count;   // por ahora guardamos el conteo

    totalCount += count;
    cCLASES++;

    eof = fgets(buffer,tBuffer,ficheroREAD);
  }

  fclose(ficheroREAD);
  ficheroREAD = 0;

  // ---- normalizar para obtener priors reales ----
  if (totalCount > 0.0)
  {
    for (int i = 0; i < cCLASES; ++i)
      priorClase[i] /= totalCount;
  }

  // Seleccionar la clase por defecto (GLOBAL) como la de mayor prior.
  // Esto es crítico en datasets muy desbalanceados (p.ej. FLARE): evita que el
  // clasificador "se escape" hacia clases minoritarias cuando la evidencia de reglas es débil.
  defaultCLASS = 0;
  double bestPrior = -1.0;
  for (int i = 0; i < cCLASES; ++i) {
    if (priorClase[i] > bestPrior) { bestPrior = priorClase[i]; defaultCLASS = i; }
  }
}


void Leer_Clasificador(char * file_clasificador)
{
  cantidadCCAR = 0;
  cantidadCCAR = (int*)malloc(sizeof(int)*cCLASES);
  totalCCAR = 0;
  totalCCAR = (int*)malloc(sizeof(int)*cCLASES);
	aveREGLA = (double**)malloc(sizeof(double*)*cCLASES);
	desvREGLA = (double**)malloc(sizeof(double*)*cCLASES);
	
	listaCCAR = (CCAR ***)malloc(sizeof(CCAR **)*cCLASES);
	
	for (int i = 0 ; i < cCLASES ; i++)
	{
		cantidadCCAR[i] = 0; totalCCAR[i] = tamanno_arregloCjt;
		listaCCAR[i] = (CCAR **)malloc(sizeof(CCAR *)*totalCCAR[i]);
		aveREGLA[i] = (double*)malloc(sizeof(double)*totalCCAR[i]);
		desvREGLA[i] = (double*)malloc(sizeof(double)*totalCCAR[i]);
	}	
	
  FILE * fichero = fopen(file_clasificador,"r");
  //buffer que almacenara lo leido
  char * buffer = 0;
  buffer = (char *)malloc(sizeof(char)*tBuffer);  
  
  char * temp = 0;
  temp = (char *)malloc(sizeof(char)*30);
  
  int inicio = 0, j = 0, elem = 0, iPOS = -1, clase = 0;
  char *eof = 0;
  int nITEMS = 0;
  CConjunto * iConjunto = 0;
  iConjunto = (CConjunto*)malloc(sizeof(CConjunto));
  iConjunto->longitud = 0;
  iConjunto->tamanno = tamanno_arregloCjt;
  iConjunto->elementos = 0;
  iConjunto->elementos = (int*)malloc(sizeof(int)*iConjunto->tamanno);
  
  eof = fgets(buffer,tBuffer,fichero);
  //int linea = 0;
  while (eof != 0)
  {
  	inicio = 0;
  	iConjunto->longitud = 0;

  	///leer la cantidad de items en la regla
  	j = 0;
  	while (buffer[inicio]!=' ')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
  	nITEMS = atoi(temp);
  	
  	for (int i = 0 ; i < nITEMS ; i++)
  	{
  		// leyendo cada item de la regla
	  	j = 0;
	  	while (buffer[inicio]!=' ')
	  	{
	  		temp[j] = buffer[inicio];
	  		inicio++; j++;
	  	}
	  	temp[j] = '\0';
  		elem = atoi(temp);
  		inicio++;
  		if (iConjunto->longitud == iConjunto->tamanno)
  		{
  			iConjunto->tamanno *= 2;
  			iConjunto->elementos = (int*)realloc(iConjunto->elementos, sizeof(int)*iConjunto->tamanno);
  		}
  		iConjunto->elementos[iConjunto->longitud] = elem;
  		iConjunto->longitud++;
  		
  	}
  	
  	// localizar la clase para insertar su CCAR
	// Soportamos reglas negativas: en el minado salen como A -> -c (equivalente a A -> neg(c)).
	int claseRaw = iConjunto->elementos[iConjunto->longitud - 1];
	unsigned char esNeg = (claseRaw < 0 ? 1 : 0);
	int claseAbs = (claseRaw < 0 ? -claseRaw : claseRaw);

	clase = claseAbs;
	iPOS = -1;
	for (int i = 0 ; i < cCLASES ; i++)
	{
		if (clasesCOLECCION[i] == claseAbs)
		{
			iPOS = i; break;
		}
	}// insertar la CCAR leida en la clase correspondiente
  	if (cantidadCCAR[iPOS] == totalCCAR[iPOS])
  	{
  		totalCCAR[iPOS] *= 2;
  		listaCCAR[iPOS] = (CCAR **)realloc(listaCCAR[iPOS], sizeof(CCAR *)*totalCCAR[iPOS]);
			aveREGLA[iPOS] = (double*)realloc(aveREGLA[iPOS], sizeof(double)*totalCCAR[iPOS]);
			desvREGLA[iPOS] = (double*)realloc(desvREGLA[iPOS], sizeof(double)*totalCCAR[iPOS]);  		
  	}
  	
  	listaCCAR[iPOS][cantidadCCAR[iPOS]] = 0;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]] = (CCAR*)malloc(sizeof(CCAR));
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->clase = clase;
	listaCCAR[iPOS][cantidadCCAR[iPOS]]->negated = esNeg;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente = (CConjunto*)malloc(sizeof(CConjunto));
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->longitud = 0;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->tamanno = nITEMS - 1;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->elementos = 0;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->elementos = (int*)malloc(sizeof(int)*listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->tamanno);
  	
  	for (int i = 0; i < nITEMS - 1 ; i++)
  	{
  		listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->elementos[listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->longitud] = iConjunto->elementos[i];
  		listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->longitud++;
  	}
  	
  	// leer los valores de soporte, y netCONF correspondientes
  	// soporte del antecedente
  	j = 0;
  	while (buffer[inicio]!=' ')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
		listaCCAR[iPOS][cantidadCCAR[iPOS]]->soporteANTECEDENTE = atof(temp); 
  	
  	// soporte del consecuente (clase)
  	j = 0;
  	while (buffer[inicio]!=' ')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
		listaCCAR[iPOS][cantidadCCAR[iPOS]]->soporteCLASE = atof(temp); 

  	// soporte de la regla
  	j = 0;
  	while (buffer[inicio]!=' ')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
		listaCCAR[iPOS][cantidadCCAR[iPOS]]->soporteREGLA = atof(temp);
		 
  	// netconf de la regla
  	j = 0;
  	while (buffer[inicio]!=' ' && buffer[inicio]!='\n')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
		listaCCAR[iPOS][cantidadCCAR[iPOS]]->netconfREGLA = atof(temp); 
		
		//actualizar el valor de aveREGLA y desvREGLA de la regla leida y de las anteriormente procesadas.
		aveREGLA[iPOS][cantidadCCAR[iPOS]] = 0;
		desvREGLA[iPOS][cantidadCCAR[iPOS]] = 0;
		
		for(int i = 0 ; i < cantidadCCAR[iPOS] ; i++)
		{
			aveREGLA[iPOS][cantidadCCAR[iPOS]] += listaCCAR[iPOS][i]->netconfREGLA;
			aveREGLA[iPOS][i] += listaCCAR[iPOS][cantidadCCAR[iPOS]]->netconfREGLA;
		}
		
		cantidadCCAR[iPOS]++;
		
	  eof = fgets(buffer,tBuffer,fichero);
  }  
  
  double mayor = 0;
  // calculando el average de cada regla de cada clase Y LA Clase mas popular
	reglasCUBREN_POS = (CConjunto**)malloc(sizeof(CConjunto*)*cCLASES);
	reglasCUBREN_NEG = (CConjunto**)malloc(sizeof(CConjunto*)*cCLASES);
	reglasCUBREN_POS_CANTIDAD = (CConjunto**)malloc(sizeof(CConjunto*)*cCLASES);

	averageWEIGHT = (double*)malloc(sizeof(double)*cCLASES);
	  
  for (int i = 0 ; i < cCLASES ; i++)
  {
  	averageWEIGHT[i] = 0;
	reglasCUBREN_POS[i] = (CConjunto*)malloc(sizeof(CConjunto));
	reglasCUBREN_POS[i]->longitud = 0;
	reglasCUBREN_POS[i]->tamanno = tamanno_arregloCjt;
	reglasCUBREN_POS[i]->elementos = (int*)malloc(sizeof(int)*reglasCUBREN_POS[i]->tamanno);

	reglasCUBREN_NEG[i] = (CConjunto*)malloc(sizeof(CConjunto));
	reglasCUBREN_NEG[i]->longitud = 0;
	reglasCUBREN_NEG[i]->tamanno = tamanno_arregloCjt;
	reglasCUBREN_NEG[i]->elementos = (int*)malloc(sizeof(int)*reglasCUBREN_NEG[i]->tamanno);
  	reglasCUBREN_POS[i]->elementos = (int*)malloc(sizeof(int)*reglasCUBREN_POS[i]->tamanno);

  	reglasCUBREN_POS_CANTIDAD[i] = 0;
  	reglasCUBREN_POS_CANTIDAD[i] = (CConjunto*)malloc(sizeof(CConjunto));
  	reglasCUBREN_POS_CANTIDAD[i]->longitud = 0;
  	reglasCUBREN_POS_CANTIDAD[i]->tamanno = tamanno_arregloCjt;
  	reglasCUBREN_POS_CANTIDAD[i]->elementos = 0;
  	reglasCUBREN_POS_CANTIDAD[i]->elementos = (int*)malloc(sizeof(int)*reglasCUBREN_POS_CANTIDAD[i]->tamanno);
  	
  	for (int j = 0 ; j < cantidadCCAR[i] ; j++)
  	{
  		aveREGLA[i][j] /= (cantidadCCAR[i] - 1);
  		averageWEIGHT[i] += listaCCAR[i][j]->netconfREGLA;
  	}
  	
  	averageWEIGHT[i] /= (1.0*cantidadCCAR[i]);
  	if (mayor < averageWEIGHT[i])
  	{
  		mayor = averageWEIGHT[i];
  		defaultCLASS = i;
  	}
  	  	
  }

  // IMPORTANTE: no usar average netconf para elegir la clase por defecto en FLARE.
  // Reforzar defaultCLASS como la clase mayoritaria (prior máximo) ya leída en Classes.dat.
  {
    int d = 0;
    double bp = -1.0;
    for (int i = 0; i < cCLASES; ++i) {
      if (priorClase[i] > bp) { bp = priorClase[i]; d = i; }
    }
    defaultCLASS = d;
  }
  
  // calculando el principio de la desviacion standard
  for (int i = 0 ; i < cCLASES ; i++)
  {
  	for (int j = 0 ; j < cantidadCCAR[i] ; j++)
  	{
  		for (int k = 0 ; k < j ; k++)
  		{
  			desvREGLA[i][j] += (listaCCAR[i][k]->netconfREGLA - aveREGLA[i][j])*(listaCCAR[i][k]->netconfREGLA - aveREGLA[i][j]);
				desvREGLA[i][k] += (listaCCAR[i][j]->netconfREGLA - aveREGLA[i][k])*(listaCCAR[i][j]->netconfREGLA - aveREGLA[i][k]);
  		}
  	}
  }

  // calculando la desviacion standard e interes de dong li
  for (int i = 0 ; i < cCLASES ; i++)
  {
  	for (int j = 0 ; j < cantidadCCAR[i] ; j++)
  	{
  		desvREGLA[i][j] = sqrt(desvREGLA[i][j]/(cantidadCCAR[i] - 1));
  		listaCCAR[i][j]->interesDONG = listaCCAR[i][j]->netconfREGLA - aveREGLA[i][j];
  		listaCCAR[i][j]->interesDONG = listaCCAR[i][j]->interesDONG - desvREGLA[i][j];
  	}
  }

	//ordenar el conjunto de reglas para cada clase
  for (int i = 0 ; i < cCLASES ; i++)
  {
  	for (int j = 0 ; j < cantidadCCAR[i] - 1 ; j++)
  	{
  		for (int k = cantidadCCAR[i] - 1 ; k > 0 ; k--)
  		{
//  			if(listaCCAR[i][k]->interesDONG > listaCCAR[i][k-1]->interesDONG)
//  			{
//  				CCAR * tmp = listaCCAR[i][k];
//  				listaCCAR[i][k] = listaCCAR[i][k-1];
//  				listaCCAR[i][k - 1] = tmp;
//  			}

  			int lenK  = listaCCAR[i][k]->antecedente->longitud;
  			int lenK1 = listaCCAR[i][k-1]->antecedente->longitud;

  			if ( (lenK > lenK1) ||
  			     (lenK == lenK1 && listaCCAR[i][k]->netconfREGLA > listaCCAR[i][k-1]->netconfREGLA) )
  			{
  			    CCAR * tmp = listaCCAR[i][k];
  			    listaCCAR[i][k] = listaCCAR[i][k-1];
  			    listaCCAR[i][k - 1] = tmp;
  			}

  		}
  	}
  }// fin de la ordenacion por clases
  
  fclose(fichero);
  fichero = 0;
  
  free(buffer);
  buffer = 0;
  free(temp);
  temp = 0;
  
  free(iConjunto->elementos);
  iConjunto->elementos = 0;

  free(iConjunto);
  iConjunto = 0;

	for (int i = 0 ; i < cCLASES ; i++)
	{
		free(aveREGLA[i]);
		aveREGLA[i] = 0;
		free(desvREGLA[i]);
		desvREGLA[i] = 0;  
	}
	
	free(aveREGLA);
	aveREGLA = 0;
	free(desvREGLA); 
	desvREGLA = 0; 
}

void Construye_ClasificadorEXACTO(char * file_clasificador,char * file_prueba, int fold_idx)
{
	// Record a small signature of the fold file so repeated runs can be matched
	// to the correct external partition generated by DivideEn10.
	long long fold_file_size = 0, fold_file_mtime = 0;
	file_signature(file_prueba, &fold_file_size, &fold_file_mtime);

	Leer_Clasificador(file_clasificador);
	printf("Clasificador cargado!!!\n");
	eficienciaEXPERIMENTO = 0;
	int aciertosCASOS = 0, totalCASOS = 0;
	int nearTieCASOS = 0;        // #instances with tieSize>1
	int negEvaluatedCASOS = 0;   // #instances where negative stage was executed (tie region)
	int negCoveredCASOS = 0;     // #instances where at least one tied class had a covering negative rule

    // --- ALPHA / EPS commented out for experiments ---
    // const double ALPHA = 1.0;   // fuerza del prior (probar 0.5, 1.0, 2.0)
    // const double EPS   = 0.0;   // margen opcional (prueba 0.01 si quieres ser más conservador)

    FILE * ficheroWRITE = fopen("AnalisisInstancias.dat","a");

    // --- NEW: lightweight CSV logs for stability analysis ---
    // One row per test instance. This enables agreement/entropy computation across repeated runs.
    const char* inst_csv_path = "stability_instance_log.csv";
    FILE* fInstCSV = fopen(inst_csv_path, "a");
    csv_write_header_if_empty(
        fInstCSV,
        inst_csv_path,
        "run_id,fold,fold_file_size,fold_file_mtime,inst_idx,true_class,pred_class,correct,default_used,near_tie,tie_size,neg_evaluated,neg_covered,best_pos,second_pos,margin,mu0_pos,pos_cover_total,neg_cover_total,chosen_pos,chosen_neg,lambda_eff");

    // One row per fold.
    const char* fold_csv_path = "stability_fold_summary.csv";
    FILE* fFoldCSV = fopen(fold_csv_path, "a");
    csv_write_header_if_empty(
        fFoldCSV,
        fold_csv_path,
        "run_id,fold,fold_file_size,fold_file_mtime,total,correct,accuracy,default_count,near_tie_count,neg_evaluated_count,neg_covered_count");

    // --- Delimitador por fold (para apendizar resultados de los 10 folds) ---
        	fprintf(ficheroWRITE, "\n==================== FOLD %d | test=%s | reglas=%s ====================\n", fold_idx, file_prueba, file_clasificador);

	// cargar cada ejemplo y clasificarlo
  FILE * fichero = fopen(file_prueba,"r");
  //buffer que almacenara lo leido
  char * buffer = 0;
  buffer = (char *)malloc(sizeof(char)*tBuffer);

  char * temp = 0;
  temp = (char *)malloc(sizeof(char)*30);

  int inicio = 0, j = 0, elem = 0, clase = 0, limite = 0;
  char *eof = 0;
  CConjunto * iConjunto = 0;
  iConjunto = (CConjunto*)malloc(sizeof(CConjunto));
  iConjunto->longitud = 0;
  iConjunto->tamanno = tamanno_arregloCjt;
  iConjunto->elementos = 0;
  iConjunto->elementos = (int*)malloc(sizeof(int)*iConjunto->tamanno);

  //double mayorAVERAGE = 0;

  int claseDEFECTO = 0;
  eof = fgets(buffer,tBuffer,fichero);
  while (eof != 0)
  {
  	inicio = 0;
  	limite = 0;
  	iConjunto->longitud = 0;
  	while (buffer[inicio] != '\n')
  	{
  		// leer el elemento de la transaccion
  		j = 0;
  		while (buffer[inicio]!= ' ' && buffer[inicio]!= '\n')
  		{
  			temp[j] = buffer[inicio];
  			inicio++; j++;
  		}
  		temp[j] = '\0';
  		elem = atoi(temp);
  		if (elem > limite) limite = elem;

  		if (iConjunto->longitud == iConjunto->tamanno)
  		{
  			iConjunto->tamanno *=2;
  			iConjunto->elementos = (int*)realloc(iConjunto->elementos, sizeof(int)*iConjunto->tamanno);
  		}

  		iConjunto->elementos[iConjunto->longitud] = elem;
  		iConjunto->longitud++;

  		if(buffer[inicio]!= '\n')
           inicio++;
  	}

  	totalCASOS++;
  	itemsEJEMPLO = (int*)malloc(sizeof(int)*limite);

  	for (int i = 0 ; i < limite ; i++)
  		itemsEJEMPLO[i] = 0;

  	for (int i = 0 ; i < iConjunto->longitud  - 1; i++)
  		itemsEJEMPLO[iConjunto->elementos[i] - 1] = 1;

  	// === PRINT INSTANCE ===
  	fprintf(ficheroWRITE, "INSTANCIA: ");
  	for (int ii = 0 ; ii < iConjunto->longitud - 1 ; ii++)
  	    fprintf(ficheroWRITE, "%d ", iConjunto->elementos[ii]);
  	fprintf(ficheroWRITE, " | Clase real = %d\n", iConjunto->elementos[iConjunto->longitud - 1]);

  	// clasificar el caso leido segun las reglas cargadas
		clase = iConjunto->elementos[iConjunto->longitud - 1];
		bool casoCUBIERTO = true;
		for (int i = 0 ; i < cCLASES ; i++)
		{
			reglasCUBREN_POS[i]->longitud = 0;
			reglasCUBREN_NEG[i]->longitud = 0;
			for (int j = 0 ; j < cantidadCCAR[i] ; j++)
			{
				casoCUBIERTO = true;
				for (int l = 0 ; l < listaCCAR[i][j]->antecedente->longitud ; l++)
				{
					if ((listaCCAR[i][j]->antecedente->elementos[l] - 1) < limite)
					{
						if (itemsEJEMPLO[listaCCAR[i][j]->antecedente->elementos[l] - 1] == 0)
						{
							casoCUBIERTO = false;
							break;
						}
					}
				}

				if (casoCUBIERTO)
				{
					// almacenar la regla que cubre el caso
					if (reglasCUBREN_POS[i]->longitud == reglasCUBREN_POS[i]->tamanno)
					{
						reglasCUBREN_POS[i]->tamanno *= 2;
						reglasCUBREN_POS[i]->elementos = (int*)realloc(reglasCUBREN_POS[i]->elementos, sizeof(int)*reglasCUBREN_POS[i]->tamanno);
					}

					if (listaCCAR[i][j]->negated) {
						reglasCUBREN_NEG[i]->elementos[reglasCUBREN_NEG[i]->longitud] = j;
						reglasCUBREN_NEG[i]->longitud++;
					} else {
						reglasCUBREN_POS[i]->elementos[reglasCUBREN_POS[i]->longitud] = j;
						reglasCUBREN_POS[i]->longitud++;
					}
				}

			}

		}

		// === PRINT RULES THAT COVER THE INSTANCE (POS and NEG with netconf) ===
		fprintf(ficheroWRITE, "Reglas que cubren la instancia (POS/NEG):\n");
		for (int c = 0; c < cCLASES; c++)
		{
		    fprintf(ficheroWRITE, "  Clase %d:\n", clasesCOLECCION[c]);

		    // Positivas
		    fprintf(ficheroWRITE, "    [+] POS: ");
		    if (reglasCUBREN_POS[c]->longitud == 0) {
		        fprintf(ficheroWRITE, "NINGUNA\n");
		    } else {
		        fprintf(ficheroWRITE, "{ ");
		        for (int r = 0; r < reglasCUBREN_POS[c]->longitud; r++) {
		            int idx = reglasCUBREN_POS[c]->elementos[r];
		            fprintf(ficheroWRITE, "#%d(", idx);
		            for (int a = 0; a < listaCCAR[c][idx]->antecedente->longitud; a++)
		                fprintf(ficheroWRITE, "%d ", listaCCAR[c][idx]->antecedente->elementos[a]);
		            fprintf(ficheroWRITE, ") [netconf=%.4f] ", listaCCAR[c][idx]->netconfREGLA);
		        }
		        fprintf(ficheroWRITE, "}\n");
		    }

		    // Negativas (A -> neg(c)) almacenadas con negated=1 y clase base = c
		    fprintf(ficheroWRITE, "    [-] NEG: ");
		    if (reglasCUBREN_NEG[c]->longitud == 0) {
		        fprintf(ficheroWRITE, "NINGUNA\n");
		    } else {
		        fprintf(ficheroWRITE, "{ ");
		        for (int r = 0; r < reglasCUBREN_NEG[c]->longitud; r++) {
		            int idx = reglasCUBREN_NEG[c]->elementos[r];
		            fprintf(ficheroWRITE, "#%d(", idx);
		            for (int a = 0; a < listaCCAR[c][idx]->antecedente->longitud; a++)
		                fprintf(ficheroWRITE, "%d ", listaCCAR[c][idx]->antecedente->elementos[a]);
		            fprintf(ficheroWRITE, ") -> neg(%d) [netconf=%.4f] ", clasesCOLECCION[c], listaCCAR[c][idx]->netconfREGLA);
		        }
		        fprintf(ficheroWRITE, "}\n");
		    }
		}
		// ==================== SELECCIÓN ESTABLE (STABLE OTF) ====================

		claseASIGNADA = -1;
		bool default_used = false;
		bool near_tie = false;
		int tie_size = 0;
		bool neg_evaluated = false;
		bool neg_covered = false;
		double best_pos = -1e300;
		double second_pos = -1e300;
		double margin = 0.0;
		double mu0_pos_logged = 0.0;
		int pos_cover_total = 0;
		int neg_cover_total = 0;
		double chosen_pos = -1e300;
		double chosen_neg = 0.0;
		double lambda_eff_chosen = 0.0;

		for (int c = 0; c < cCLASES; ++c) {
			pos_cover_total += reglasCUBREN_POS[c]->longitud;
			neg_cover_total += reglasCUBREN_NEG[c]->longitud;
		}

		// 1) Si ninguna clase cubre esta instancia, usar DEFAULT
		int clasesConCobertura = 0;
		for (int c = 0; c < cCLASES; ++c)
		    if (reglasCUBREN_POS[c]->longitud > 0) clasesConCobertura++;

		if (clasesConCobertura == 0) {
		    claseASIGNADA = clasesCOLECCION[defaultCLASS];
		    default_used = true;
		    claseDEFECTO++;
		} else {
		    // 2) mu0_pos global: promedio de netconf de TODAS las reglas POSITIVAS que cubren esta instancia
		    //    (si el ejemplo está poco cubierto, esto evita que el score dependa de 1 sola regla)
		    double mu0_pos = 0.0;
		    int cnt0 = 0;
		    for (int c = 0; c < cCLASES; ++c) {
		        for (int r = 0; r < reglasCUBREN_POS[c]->longitud; ++r) {
		            int idx = reglasCUBREN_POS[c]->elementos[r];
		            mu0_pos += listaCCAR[c][idx]->netconfREGLA;
		            cnt0++;
		        }
		    }
		    if (cnt0 > 0) mu0_pos /= (double)cnt0;
		    mu0_pos_logged = mu0_pos;

		    // 3) Selección en 2 etapas:
		    //    - Primero decidimos SOLO con evidencia positiva (score estable POS).
		    //    - SOLO si hay empate por evidencia positiva (dentro de STABLE_TIE_EPS),
		    //      usamos CARs negativas para deshacer el empate.
		    //    Esto evita que la evidencia negativa "mate" la clase correcta cuando
		    //    ya hay un ganador claro por soporte/positivas.
		    double *posScore = (double*)malloc(sizeof(double) * cCLASES);
		    for (int c = 0; c < cCLASES; ++c) posScore[c] = -1e300;

		    double bestPos = -1e300;
		    int bestPosC = -1;
		    // Base score includes a small prior term to avoid over-predicting minority classes in highly imbalanced datasets (e.g., FLARE).
		    double bestBase = -1e300;
		    int bestBaseC = -1;
		    for (int c = 0; c < cCLASES; ++c) {
		        double p = stable_score_pos_class(c, reglasCUBREN_POS, listaCCAR, mu0_pos);
		        posScore[c] = p;
		        if (p <= -1e200) continue;
		        // Track pure-positive winner (for near-tie detection)

		        // Find the true maximum by positives (do NOT use the behavioral tie epsilon here).
		        if (bestPosC == -1 || p > bestPos + STABLE_NUM_EPS) {
		            bestPos = p;
		            bestPosC = c;
		        } else if (fabs(p - bestPos) <= STABLE_NUM_EPS) {
		            // desempate determinista por id de clase (solo para fijar bestPosC)
		            if (clasesCOLECCION[c] < clasesCOLECCION[bestPosC]) bestPosC = c;
		        }

		        // Track base-score winner (positives + small prior term) for the final decision when there is a clear winner by positives.
		        double base = p + STABLE_LAMBDA_PRIOR * log(priorClase[c]);
		        if (bestBaseC == -1 || base > bestBase + STABLE_NUM_EPS) {
		            bestBase = base;
		            bestBaseC = c;
		        } else if (fabs(base - bestBase) <= STABLE_NUM_EPS) {
		            if (clasesCOLECCION[c] < clasesCOLECCION[bestBaseC]) bestBaseC = c;
		        }
		    }

		    // Track the second-best positive score (for margin / ambiguity diagnostics)
		    second_pos = -1e300;
		    for (int c = 0; c < cCLASES; ++c) {
		        if (posScore[c] <= -1e200) continue;
		        if (c == bestPosC) continue;
		        if (second_pos <= -1e200 || posScore[c] > second_pos + STABLE_NUM_EPS) {
		            second_pos = posScore[c];
		        }
		    }
		    best_pos = bestPos;
		    if (second_pos > -1e200) margin = bestPos - second_pos;

		    if (bestPosC == -1) {
		        free(posScore);
		        claseASIGNADA = clasesCOLECCION[defaultCLASS];
		        default_used = true;
		        claseDEFECTO++;
		    } else {

		        // Construir conjunto de clases empatadas por evidencia positiva
		        int tieCount = 0;
		        for (int c = 0; c < cCLASES; ++c) {
		            if (posScore[c] <= -1e200) continue;
		            // Near-tie set: classes whose positive score is close to the best.
		            if ((bestPos - posScore[c]) <= STABLE_TIE_EPS) tieCount++;
		        }
		        tie_size = tieCount;
		        near_tie = (tieCount > 1);

		        if (tieCount <= 1) {
		            // Ganador claro por positivas: IGNORAMOS negativas
		            claseASIGNADA = clasesCOLECCION[bestBaseC];
		            chosen_pos = posScore[bestBaseC];
		            free(posScore);
		        } else {
		            neg_evaluated = true;
		            // Empate por positivas: usar negativas SOLO dentro del conjunto empatado
		            double bestScore = -1e300;
		            double bestNeg = 1e300;
		            int bestC = -1;
		            for (int c = 0; c < cCLASES; ++c) {
		                if (posScore[c] <= -1e200) continue;
		                if ((bestPos - posScore[c]) > STABLE_TIE_EPS) continue;
		                if (reglasCUBREN_NEG[c]->longitud > 0) neg_covered = true;

		                // Adaptive negative weight: negatives matter most when positives are extremely close.
		                double delta = bestPos - posScore[c];
		                if (delta < 0.0) delta = 0.0;
		                double lambda_eff = STABLE_LAMBDA_NEG * exp(-delta / STABLE_TAU);
		
		                double negSc = stable_score_neg_class(c, reglasCUBREN_NEG, listaCCAR);
		                double sc = posScore[c] - (lambda_eff * negSc) + STABLE_LAMBDA_PRIOR * log(priorClase[c]);
		
		                if (bestC == -1 || sc > bestScore + STABLE_NUM_EPS) {
		                    bestScore = sc;
		                    bestNeg = negSc;
		                    bestC = c;
		                } else if (fabs(sc - bestScore) < STABLE_NUM_EPS) {
		                    // desempate: menor penalización (menos evidencia en contra)
		                    if (negSc < bestNeg - STABLE_NUM_EPS) {
		                        bestNeg = negSc;
		                        bestC = c;
		                    } else if (fabs(negSc - bestNeg) < STABLE_NUM_EPS) {
		                        if (clasesCOLECCION[c] < clasesCOLECCION[bestC]) bestC = c;
		                    }
		                }
		            }
		
		            if (bestC == -1) {
		                claseASIGNADA = clasesCOLECCION[defaultCLASS];
		                default_used = true;
		                claseDEFECTO++;
		            } else {
		                claseASIGNADA = clasesCOLECCION[bestC];
		                chosen_pos = posScore[bestC];
		                // Log the chosen negative evidence and lambda used for the winning class
		                double delta = bestPos - posScore[bestC];
		                if (delta < 0.0) delta = 0.0;
		                lambda_eff_chosen = STABLE_LAMBDA_NEG * exp(-delta / STABLE_TAU);
		                chosen_neg = stable_score_neg_class(bestC, reglasCUBREN_NEG, listaCCAR);
		            }
		            free(posScore);
		        }
		    }

		}

  	// chequear clasificacion realizada con la clase que traia el ejemplo
	int correct = (clase == claseASIGNADA) ? 1 : 0;
	if (correct) aciertosCASOS++;

	// Update per-fold stability counters
	if (near_tie) nearTieCASOS++;
	if (neg_evaluated) negEvaluatedCASOS++;
	if (neg_covered) negCoveredCASOS++;

	// Write lightweight instance row (CSV)
	if (fInstCSV) {
	    fprintf(fInstCSV,
	        "%d,%d,%lld,%lld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.10g,%.10g,%.10g,%.10g,%d,%d,%.10g,%.10g,%.10g\n",
	        STABILITY_RUN_ID,
	        fold_idx,
	        fold_file_size,
	        fold_file_mtime,
	        totalCASOS,
	        clase,
	        claseASIGNADA,
	        correct,
	        default_used ? 1 : 0,
	        near_tie ? 1 : 0,
	        tie_size,
	        neg_evaluated ? 1 : 0,
	        neg_covered ? 1 : 0,
	        best_pos,
	        second_pos,
	        margin,
	        mu0_pos_logged,
	        pos_cover_total,
	        neg_cover_total,
	        chosen_pos,
	        chosen_neg,
	        lambda_eff_chosen);
	}
  	fprintf(ficheroWRITE, "Clase = %i, ASIGNADA = %i\n", clase,claseASIGNADA);
  	fprintf(ficheroWRITE, "Aciertos = %i de Total = %i\n",aciertosCASOS, totalCASOS);
  	fprintf(ficheroWRITE, "Casos clasificador con clase por defecto = %i\n",claseDEFECTO);

  	eof = fgets(buffer,tBuffer,fichero);
  	free(itemsEJEMPLO);
  	itemsEJEMPLO = 0;
  }
	// cargar cada ejemplo y clasificarlo

	// calcular eficacia de la clasificacion del grupo de ejemplos
	eficienciaEXPERIMENTO = (aciertosCASOS*100.0)/totalCASOS;

	// Write fold summary row (CSV)
	if (fFoldCSV) {
	    fprintf(fFoldCSV, "%d,%d,%lld,%lld,%d,%d,%.6f,%d,%d,%d,%d\n",
	        STABILITY_RUN_ID,
	        fold_idx,
	        fold_file_size,
	        fold_file_mtime,
	        totalCASOS,
	        aciertosCASOS,
	        eficienciaEXPERIMENTO,
	        claseDEFECTO,
	        nearTieCASOS,
	        negEvaluatedCASOS,
	        negCoveredCASOS);
	    fflush(fFoldCSV);
	}
	if (fInstCSV) { fflush(fInstCSV); fclose(fInstCSV); fInstCSV = NULL; }
	if (fFoldCSV) { fclose(fFoldCSV); fFoldCSV = NULL; }

	for (int i = 0 ; i < cCLASES ; i++)
	{
		free(reglasCUBREN_POS[i]->elementos);
		reglasCUBREN_POS[i]->elementos = 0;
		free(reglasCUBREN_POS[i]);
		reglasCUBREN_POS[i] = 0;

		free(reglasCUBREN_POS_CANTIDAD[i]->elementos);
		reglasCUBREN_POS_CANTIDAD[i]->elementos = 0;
		free(reglasCUBREN_POS_CANTIDAD[i]);
		reglasCUBREN_POS_CANTIDAD[i] = 0;
		for (int j = 0 ; j < cantidadCCAR[i] ; j++)
		{
			free(listaCCAR[i][j]->antecedente->elementos);
			listaCCAR[i][j]->antecedente->elementos = 0;
			free(listaCCAR[i][j]->antecedente);
			listaCCAR[i][j]->antecedente = 0;
			free(listaCCAR[i][j]);
			listaCCAR[i][j] = 0;

		}

		free(listaCCAR[i]);
		listaCCAR[i] = 0;

	}

	free(reglasCUBREN_POS);
	reglasCUBREN_POS = 0;
	free(reglasCUBREN_POS_CANTIDAD);
	reglasCUBREN_POS_CANTIDAD = 0;
	free(listaCCAR);
	listaCCAR = 0;

	free(cantidadCCAR);
	cantidadCCAR = 0;
	free(totalCCAR);
	totalCCAR = 0;

	free(averageWEIGHT);
	averageWEIGHT = 0;

	fclose(fichero);
	fichero = 0;
	fclose(ficheroWRITE);
	ficheroWRITE = 0;
	free(buffer);
	buffer = 0;
	free(temp);
	temp = 0;

	free(iConjunto->elementos);
	iConjunto->elementos = 0;
	free(iConjunto);
	iConjunto = 0;
}

void Clasifica_Dataset(int argc, char *argv[])
{
  printf("Iniciando...\n");	
  Leer_File_CLASSES();
	eficienciaPROMEDIO = 0;
  printf("readed...\n");

	nombres_files = (char**)malloc(sizeof(char*)*10);
	nombres_REGLAS = (char**)malloc(sizeof(char*)*10);

  FILE * ficheroWRITE = 0;
  ficheroWRITE = fopen(argv[1],"w");
	
	for (int i = 0 ; i < 10 ; i++)
	{
  	nombres_files[i] = (char*)malloc(sizeof(char)*10);
  	sprintf(nombres_files[i],"%i.dat",i+1);
		nombres_REGLAS[i] = (char*)malloc(sizeof(char)*20);
  	sprintf(nombres_REGLAS[i],"ReglasDataset%i.dat",i+1);		

		Construye_ClasificadorEXACTO(nombres_REGLAS[i],nombres_files[i],i+1);
		eficienciaPROMEDIO += eficienciaEXPERIMENTO;
		
		fprintf(ficheroWRITE, "%s\n",nombres_files[i]);
		fprintf(ficheroWRITE, "Eficiencia obtenida = %.4f\n\n",eficienciaEXPERIMENTO);
		fflush(ficheroWRITE);
		printf("Eficiencia obtenida = %.4f\n\n",eficienciaEXPERIMENTO);
		
		
	}
	printf("Clasificado dataset!!\n");
	
	eficienciaPROMEDIO /= 10.0;
	fprintf(ficheroWRITE, "Eficiencia promedio del proceso = %.4f\n", eficienciaPROMEDIO);
	printf("Eficiencia promedio del proceso = %.4f\n", eficienciaPROMEDIO);
	fflush(ficheroWRITE);
	fclose(ficheroWRITE);
	ficheroWRITE = 0;
	fprintf(ficheroWRITE, "Eficiencia promedio obtenida = %.4f\n",eficienciaPROMEDIO);
	fclose(ficheroWRITE);
	ficheroWRITE = 0;
	
	for (int i = 0 ; i < 10 ; i++)
	{
		free(nombres_files[i]);
		nombres_files[i] = 0;
		free(nombres_REGLAS[i]);
		nombres_REGLAS[i] = 0;
	}
	free(nombres_files);
	nombres_files = 0;
	free(nombres_REGLAS);
	nombres_REGLAS = 0;

	free(clasesCOLECCION);
	clasesCOLECCION = 0;

    free(priorClase);
    priorClase = 0;
}

// implementacion de los metodos
int main(int argc, char *argv[])
{
  if (process_arguments(argc,argv))
    return 1;
  
	Clasifica_Dataset(argc,argv);
  return EXIT_SUCCESS;
}
