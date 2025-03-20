<script lang="ts">
  import { onMount } from "svelte";

  let images: string[] = []
  let closest_images: string[] = []
  let query: string = ""

  async function getImages() {
    const response = await fetch("http://localhost:8080/imgpaths");
    const data = await response.json()
    images = data
  }
  async function search_images() {
    closest_images = []
    const response = await fetch(`http://localhost:8080/search?query=${query}`)
    let closest_matches = await response.json()
    closest_images = closest_matches
  }
  onMount(async () => {
    await getImages()
  })
</script>

<div class=" bg-amber-500">
  <h1 class=" text-2xl">CLOSEST</h1>
  {#each closest_images as closest_image}
    <img style="width: 250px;" src={`http://localhost:8080/${closest_image}`} alt="oopsie">
  {/each}
</div>

<div>
  <div class="flex flex-row h-12 p-1 bg-[#343434] items-center justify-center fixed top-0 w-screen">
    <input bind:value={query} class=" h-full w-[50%] bg-[#444444] p-1 rounded-xl outline-0" type="text" placeholder="Search...">
    <button onclick={search_images} >search</button>
  </div>

  <div class=" mt-12">
    {#each images as image}
      <img style="width: 250px;" src={`http://localhost:8080/images/${image}`} alt="uh oh">
    {/each}
  </div>
</div>
