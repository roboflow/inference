---
hide:
  - toc
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.8/purify.min.js"></script>
<link rel="stylesheet" href="/styles/workflows.css">

<script src="https://kit.fontawesome.com/c9c3956d39.js" crossorigin="anonymous"></script>


<section class="mdx-container portfolio-section">
  <div class="md-grid md-typeset">
    <div class="text-center">
    {% if custom_title %}
      <h1>{{ title }}</h1>
      <p style="white-space: pre-wrap;">{{ description.strip() }}</p>
    {% else %}
      <h1>Workflow Blocks</h1>
      <p>Workflows are made of Blocks. Blocks can be connected to build multi-step computer vision applications.</p>
      <p>Below is a list of all the Blocks supported in Workflows.</p>
    {% endif %}
    </div>

      {% for section in block_sections %}
      <div class="section">
        <h2>{{ section.title }}</h2>
        <div class="blocks">
            <div class="custom-grid">
            {% for block in blocks_by_section[section.id] %}



            <a href="{{block.url}}" style="border: 1px solid {{ block.color }}; border-radius: 0.25rem;">
              <div class="block">
                
                <div class="block_name"> <i class="{{block.icon}}" style="color: {{ block.color }}; fill: {{ block.color }}"> </i>  {{ block.name }}</div>
                <div class="block_description">{{ block.description }}</div>
                <!-- <div class="block_license">{{ block.license }}</div> -->
              </div>
            </a>


          
          {% endfor %}
          </div>
        </div> 
      </div>
      {% endfor %}
    </div>
  </div>
</section>

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}

.block {
  border-radius: 4px;
  padding: 10px;
  height: 100%;
}

.block_name {
  font-size: large;
  color: black;
}

.block_description {
    padding-top: 5px;
    color: #444;
}

.block_license {
  background-color: #14b8a6; 
  color: #fff; 
  padding: 2px 4px;
  border-radius: 4px; 
  font-size: small;
}


</style>