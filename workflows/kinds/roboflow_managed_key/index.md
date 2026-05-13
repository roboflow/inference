
# `ROBOFLOW_MANAGED_KEY` Kind

Roboflow-managed key or credential

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


This kind represents a key or credential managed by Roboflow or stored in your Roboflow account.
It can be used for for third party APIs that Roboflow can proxy requests on behalf of the user.

If set to ``rf_key:account`` third party api calls will use Roboflow owned API key and proxied requests will be charged corosponding credits.  You can set the value to ``rf_key:user:<key_id>`` where key_id references a third party key you have stored in your Roboflow settigns.


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
