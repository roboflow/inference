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
      <h1>Workflow Blocks</h1>
    </div>
    
      {% for section in block_sections %}
      <div class="section">
        <h2>{{ section.title | capitalize }}</h2>
        <div class="blocks">
            <div class="custom-grid">
            {% for block in blocks_by_section[section.id] %}



            <a href="{{block.url}}">
              <div class="block">
                
                <div class="block_name"> <i class="{{block.icon}}" > </i>  {{ block.name }}</div>
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
  border: 1px solid black;
  border-radius: 4px;
  padding: 10px;
  height: 100px;
}

.block_name {
  font-size: large;
  color: black;
}

.block_description {
    font-size: 0.65em;
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