# CLIP Image Vectorizer with BentoML

This project provides an API for vectorizing images using OpenAI's CLIP model using BentoML. It allows you to send images to the API and receive a vector representation that encodes meaningful features of the image.

The processor automatically resizes images to the model's expected input dimensions (e.g., 224x224 for this model) and normalizes pixel values.

## Features
- Easy-to-use REST API for image vectorization.
- Powered by the OpenAI CLIP model.
- Serves vectorization directly from BentoML.
- Ready for local testing.

## Supported Image Formats

The service supports the following image formats based on Pillow's capabilities:

| **Format**      | **File Extensions**         | **Description**                                                              |
|------------------|-----------------------------|------------------------------------------------------------------------------|
| **JPEG**        | `.jpg`, `.jpeg`, `.jpe`     | Common format with lossy compression, widely used for photographs.          |
| **PNG**         | `.png`                      | Lossless compression, supports transparency (alpha channel).                |
| **BMP**         | `.bmp`, `.dib`              | Bitmap image format, uncompressed.                                          |
| **GIF**         | `.gif`                      | Supports animation and transparency; only the first frame is processed.     |
| **TIFF**        | `.tiff`, `.tif`             | Flexible format supporting multiple layers and compression options.         |
| **PPM**         | `.ppm`, `.pgm`, `.pbm`      | Portable Pixmap formats (NetPBM).                                           |
| **ICO**         | `.ico`                      | Icon format, often used for application icons.                              |
| **WEBP**        | `.webp`                     | Modern image format for web usage, supports both lossy and lossless modes.  |
| **DDS**         | `.dds`                      | DirectDraw Surface, used for textures in graphics applications.             |
| **TGA**         | `.tga`                      | Targa format, often used in video games and graphics.                       |
| **HDR**         | `.hdr`                      | High Dynamic Range image format, used for realistic lighting.               |
| **JPEG 2000**   | `.jp2`, `.j2k`, `.jpx`      | Advanced JPEG format with better compression.                               |


---

## Local install

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python3 -m pip install transformers Pillow torch bentoml
   ```

---

## Run a Local Service

1. Start the BentoML service:
   ```bash
   bentoml serve service:svc
   ```
2. Verify the service is running:
   - Health Check:
     ```bash
     curl -v http://127.0.0.1:3000/livez
     ```
   - Metrics:
     ```bash
     curl http://127.0.0.1:3000/metrics
     ```

---

## Vectorize an Image

Send an image to the API and receive a vector representation:

```bash
curl -X POST -H "Content-Type: image/jpeg" --data-binary @image.jpg http://127.0.0.1:3000/vectorize
```

The response will be a JSON object containing the image vector:

```
{"vector":[-0.6687430143356323,-0.022710340097546577,-0.15384919941425323,-0.2727549970149994,0.4632737934589386,0.16514046490192413,0.09627659618854523,0.6447893977165222,0.4654671251773834,0.02930556796491146,-0.020234737545251846,0.13032759726047516,-0.542938232421875,-0.6593307256698608,-0.18386724591255188,0.07420549541711807,-0.020180068910121918,-0.13537637889385223,0.1709609478712082,-0.28379443287849426,-0.09426041692495346,0.3305407166481018,-0.4156511723995209,0.17745189368724823,0.3153534233570099,-0.02357916720211506,0.35171350836753845,0.2296837866306305,0.8345032930374146,-0.2707275450229645,0.19283245503902435,-0.4151032269001007,-0.44615352153778076,-0.3210270404815674,0.3318961560726166,0.4676361083984375,-0.4021487832069397,0.23209887742996216,-0.3185116648674011,0.7966142296791077,-0.2598493993282318,0.35262978076934814,0.3129599690437317,-0.17915745079517365,0.28403380513191223,-1.9820750951766968,-0.008681900799274445,0.39878207445144653,-0.4686589539051056,0.25742441415786743,-0.14648295938968658,-0.1207902729511261,0.08937818557024002,-0.25353747606277466,-0.536446213722229,0.12080903351306915,0.3860810399055481,-0.3868846595287323,-0.1369106024503708,-0.3076460361480713,0.4234112799167633,-0.397183895111084,-0.09535045921802521,0.03087746538221836,0.02453356608748436,0.08965231478214264,-0.7286188006401062,0.7265434861183167,-0.12212776392698288,-0.32552242279052734,-0.4765230119228363,-0.24616089463233948,0.16469888389110565,-0.017333701252937317,-0.6357914805412292,-0.06779493391513824,-0.14376555383205414,0.5299485921859741,0.17236435413360596,-0.40594810247421265,-0.259965717792511,0.2835208773612976,-0.08705392479896545,0.07975983619689941,0.24842654168605804,0.08793193101882935,0.07566756755113602,-0.05389964580535889,0.50528484582901,-0.33495745062828064,-0.2059670239686966,-0.023608390241861343,-7.187380790710449,0.10389988869428635,-0.4126972556114197,0.11890425533056259,0.1492355465888977,-0.4540826678276062,-0.5568014979362488,-0.1283746063709259,0.1250491589307785,0.3253483772277832,0.11105281859636307,0.09786208719015121,0.19193975627422333,-0.15640385448932648,-1.1143189668655396,0.19040577113628387,-0.1052761971950531,-0.013899981044232845,0.21917814016342163,0.6362742781639099,-0.24634525179862976,-0.0022915152367204428,0.07358803600072861,0.060874905437231064,-0.12489165365695953,0.2864362895488739,0.2366129755973816,-0.36690008640289307,-0.08417553454637527,-0.41571661829948425,0.0265496876090765,-0.06496751308441162,0.18957103788852692,-0.012186618521809578,0.2702372968196869,0.3300398886203766,0.23428553342819214,0.12994609773159027,-0.21896624565124512,-0.5408898591995239,-0.12862853705883026,0.8993349075317383,-0.10929622501134872,-0.16828884184360504,-0.05212879180908203,-0.39806613326072693,-0.28135260939598083,-0.049532048404216766,-0.18422460556030273,0.24449452757835388,-0.42220720648765564,0.041131943464279175,-0.2257036417722702,-0.20136742293834686,-0.2935275733470917,-0.35853374004364014,-0.32096046209335327,0.11864367127418518,0.49388977885246277,-0.15340092778205872,0.28780031204223633,-0.22219887375831604,0.47611895203590393,-0.07730628550052643,-0.2743844985961914,-0.2890458405017853,-0.04988916218280792,-0.22355858981609344,0.12299978733062744,-0.26488932967185974,-0.18939678370952606,-0.1037522554397583,-0.17824573814868927,-0.0656704232096672,0.3611837327480316,0.5910423994064331,0.23946894705295563,-0.018466074019670486,-0.2885538637638092,-0.011474087834358215,0.1443667709827423,-0.04336171969771385,0.05861891433596611,0.08690109103918076,-0.38174137473106384,-0.03723884001374245,-0.9623693823814392,0.20358490943908691,-0.13544803857803345,0.010264777578413486,0.13430726528167725,-0.09786717593669891,0.1561468094587326,0.46919846534729004,-0.498098224401474,-0.16015055775642395,0.33450761437416077,-0.4248847961425781,0.2797245383262634,-0.0552232451736927,0.023255901411175728,0.4122523367404938,0.001955569488927722,-0.28531312942504883,-0.14753109216690063,0.17971429228782654,-0.010966102592647076,0.5021719932556152,0.05363306403160095,0.6847209334373474,-0.2836933135986328,0.5522644519805908,-0.016495738178491592,-0.45481076836586,-0.22258462011814117,-0.10553868114948273,0.08228834718465805,0.23540951311588287,-0.6100614666938782,0.4537906050682068,-0.512470006942749,-0.3049924671649933,0.15360650420188904,-0.03903038799762726,0.552315890789032,0.49156108498573303,-0.07450329512357712,-0.24077649414539337,-0.04143289104104042,-0.13271500170230865,-0.05301252380013466,0.3722735345363617,-0.15448760986328125,0.07515329122543335,-0.0459660179913044,-0.290149986743927,0.13526467978954315,0.03725915402173996,0.1986769586801529,0.17600400745868683,0.19316820800304413,-0.47234046459198,-0.13859936594963074,-1.0071736574172974,-0.3022332191467285,0.29640182852745056,0.11855960637331009,0.07150129228830338,0.5902463793754578,-0.25115805864334106,-0.21826037764549255,0.35356080532073975,-0.4266735017299652,-0.4658615291118622,-0.008019642904400826,0.5278207063674927,0.027374403551220894,-0.2268766313791275,-0.12756898999214172,0.545117199420929,0.07765114307403564,0.567132294178009,-0.3175605237483978,0.1553553342819214,0.0224529467523098,-0.2392469346523285,0.9517398476600647,0.29493796825408936,-0.257263720035553,0.30815431475639343,-0.2926419973373413,-0.42541107535362244,-0.13207559287548065,-0.26433080434799194,-0.24864697456359863,0.3418199419975281,0.10647514462471008,-0.2173406034708023,-0.6599851250648499,0.11561860889196396,0.19037136435508728,-0.252069890499115,0.24623925983905792,-0.11892348527908325,-0.34424832463264465,0.2680620551109314,-0.26266348361968994,0.0652148425579071,0.17669381201267242,0.48943397402763367,-0.8575471639633179,-0.44590499997138977,-0.08370854705572128,-0.07729998975992203,1.5670162439346313,0.006032233126461506,0.1090606302022934,-0.168703094124794,0.04758869856595993,-0.624056875705719,0.1192607507109642,-0.07241025567054749,0.3471008241176605,-0.07623223960399628,-0.7778768539428711,0.43746113777160645,-0.0544099360704422,0.07533494383096695,0.24460424482822418,-0.3199358284473419,0.381843239068985,-0.010244235396385193,0.14476929605007172,0.06357446312904358,-0.09056402742862701,0.16840197145938873,-0.3813087046146393,0.005495330318808556,0.8527001142501831,-0.5172562003135681,0.8982032537460327,-0.3579867482185364,-0.05650828778743744,0.2134462296962738,0.17752377688884735,0.4655456244945526,-0.22543883323669434,-0.8672400116920471,0.4614712595939636,-0.7720849514007568,-0.24479277431964874,0.2790658175945282,-0.275779664516449,-0.5715225338935852,-0.08155003190040588,0.04851926863193512,-0.0786409005522728,0.2547242045402527,-0.02729002758860588,-0.4497646391391754,-0.03301505744457245,-0.36874547600746155,0.07217635959386826,-0.1605895608663559,0.45290327072143555,0.0006251371232792735,-0.05712286755442619,-0.2005710005760193,0.1854095458984375,-0.5675923824310303,0.1768413484096527,0.2925603687763214,0.062460485845804214,0.3148953318595886,0.07779975235462189,0.40383946895599365,-0.2845822870731354,-0.41638797521591187,0.1070960983633995,0.22801965475082397,0.2413344532251358,0.13489122688770294,0.019333994016051292,-0.7348327040672302,-0.039956312626600266,-0.9361084699630737,0.1517840474843979,-0.17587532103061676,-0.7749672532081604,-0.10122931003570557,-0.07806580513715744,-0.3306531012058258,-0.5295395851135254,-0.2496688812971115,0.13395492732524872,-0.9126725196838379,0.33107060194015503,0.32873207330703735,-0.4005899429321289,0.18116572499275208,-0.005266837775707245,-0.28336676955223083,0.4978158473968506,-0.2628476321697235,0.6784069538116455,0.7480951547622681,-0.323691725730896,0.4310588836669922,-0.3866272568702698,0.7127205729484558,-0.0032884920947253704,0.14292868971824646,-0.19927169382572174,0.07031934708356857,0.3064906597137451,0.3166056275367737,-0.2888231575489044,-1.8909797668457031,-0.7489597797393799,-0.13859893381595612,-0.008961755782365799,-0.5116487145423889,-0.11851406842470169,-0.16428516805171967,0.18195262551307678,-0.2641390264034271,-0.6203307509422302,0.033755987882614136,0.04241091012954712,-0.07677076756954193,0.6833506226539612,0.05944281816482544,0.5599114298820496,-0.10904783755540848,0.018559250980615616,0.1960470825433731,0.3265683352947235,-0.15698853135108948,-0.07001422345638275,-0.3610921800136566,-0.08147881925106049,-0.06022673472762108,-0.3509008586406708,-0.3838540315628052,-0.14048559963703156,0.20279240608215332,0.1394013613462448,0.134286567568779,0.16770517826080322,0.11384274065494537,0.28905630111694336,-0.35054340958595276,-0.23329775035381317,-0.3413116931915283,-0.19930826127529144,0.03135235980153084,-1.178224802017212,0.2229481190443039,0.01727207377552986,-0.16495048999786377,0.6941332817077637,0.23341211676597595,0.11046919226646423,-0.2574443221092224,0.03829745948314667,0.005965203512459993,0.013002031482756138,-0.4964532256126404,0.02570381760597229,-0.3137991428375244,0.20112603902816772,-0.3890151083469391,-0.19029372930526733,0.6843201518058777,0.3852490186691284,-0.045351527631282806,0.020188722759485245,-0.08155635744333267,0.5119767785072327,0.2027740180492401,-0.01853560097515583,-0.3524211645126343,-0.06180796027183533,0.16514016687870026,0.3698502480983734,-0.05076222866773605,-0.1643448919057846,-0.025796962901949883,-0.37968218326568604,0.08392167836427689,0.5584202408790588,-0.04495157301425934,-0.19844365119934082,-0.19693666696548462,0.1520376354455948,-0.2164839208126068,-0.548904538154602,-0.5461486577987671,-0.47394636273384094,-0.07374518364667892,0.15879611670970917,-0.11984585970640182,0.21159540116786957,-0.5362032651901245,-0.05171551555395126,0.23623153567314148,-0.07471104711294174,0.2319726198911667,0.10769246518611908,-0.1358712911605835,-0.22569306194782257,0.38241857290267944,-0.03833075985312462,-0.5425991415977478,-0.09216589480638504,0.24640345573425293,0.009783736430108547,-0.05307791754603386,0.01857549138367176,-0.26878634095191956,0.1928669512271881,0.11005572229623795,-0.24909573793411255,0.2915172278881073,0.4965748190879822,-0.002311174990609288,-0.26480889320373535,0.4019930064678192,-0.5284759402275085,-0.19770491123199463,-0.003169749630615115,0.418404757976532,0.33907899260520935,0.22131070494651794,0.06638256460428238,0.370095819234848,-0.03581619635224342,0.2209053784608841,-0.006727532483637333,0.10887718945741653]}
```

## Build a BentoML container image

```
bentoml build --containerize
```

### Get the tag name

To start up a container you'll need the container tag that was just created. Use the following command to list tags:

```
bentoml list
```

You'll see an output similar to:

```
 Tag                                     Size       Model Size  Creation Time
 clip_image_vectorizer:e4kep3vx56ru3mg4  18.44 KiB  0.00 B      2024-12-11 18:38:46
 ```

### Run the container

```
docker run --rm -p 3000:3000 clip_image_vectorizer:TAG_NAME
```