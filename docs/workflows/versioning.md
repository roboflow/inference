# Workflows versioning

Understanding the life-cycle of Workflows ecosystem is an important topic, 
especially from blocks developer perspective. Those are rules that apply:

* Workflows is part of `inference` - the package itself has a release whenever 
any of its component changes and those changes are ready to be published

* Workflows Execution Engine declares it's version. The game plan is the following:

    * core of workflows is capable to host multiple versions of Execution Engine - 
    for instance current stable version and development version

    * stable version is maintained and new features are added until there is 
    need for new version and the new version is accepted
    
    * since new version is fully operational, previously stable version starts 
    being deprecated - there is grace period when old version will be patched 
    with bug fixes (but new features will not be added), after that it will
    be left as is. During grace period we will call blocks creators to upgrade 
    their plugins according to requirements of new version

    * core library only maintains single Execution Engine version for each major -
    making a promise that features within major will be non-breaking and Workflow 
    created under version `1.0.0` will be fully functional under version `1.4.3` of 
    Execution Engine

* to ensure stability of the ecosystem over time:
    
    * Each Workflow Definition declares Execution Engine version it is compatible with. 
    Since the core library only maintains single version for Execution Engine, 
    `version: 1.1.0` in Workflow Definition actually request Execution Engine in version
    `>=1.1.0,<2.0.0`

    * Each block, in its manifest should provide reasonable Execution Engine compatibility -
    for instance - if block rely on Execution Engine feature introduced in `1.3.7` it should 
    specify `>=1.3.7,<2.0.0` as compatible versions of Engine

* Workflows blocks may be optionally versioned (which we recommend and apply for Roboflow plugins).

    * we propose the following naming convention for blocks' type identifiers: 
    `{plugin_name}/{block_family_name}@v{X}` to ensure good utilisation of blocks identifier 
    namespace

    * we suggest to only modify specific version of the block if bug-fix is needed, 
    all other changes to block should yield new version

    * each version of the block is to be submitted into new module
    (as suggested [here](/workflows/blocks_bundling.md)) - even **for the price of code duplication**
    as we think stability is more important than DRY in this particular case

    * on the similar note, we suggest each block to be as independent as possible, 
    as code which is shared across blocks, may unintentionally modify other blocks 
    destroying the stability of your Workflows
