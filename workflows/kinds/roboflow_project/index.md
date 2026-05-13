
# `roboflow_project` Kind

Roboflow project name

## Data representation



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `str`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `str`

## Details


This kind represents value specific for Roboflow platform. At the platform, each project has 
unique name and the value behind this kind represent this name. To learn more 
on how to kick-off with Roboflow project - visit [this](https://blog.roboflow.com/getting-started-with-roboflow/) page. 


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
