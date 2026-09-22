import { defineConfig } from 'vitepress';

const guideSidebar = [
  {
    text: 'Guide',
    items: [
      { text: 'Quick Start', link: '/guide/quick-start' },
      { text: 'Installation', link: '/guide/installation' },
    ],
  },
];

const usageSidebar = [
  {
    text: 'Usage',
    items: [
      { text: 'Data Structure', link: '/usage/data-structure' },
      { text: 'Lifted Datasets', link: '/usage/lifted-datasets' },
      { text: 'Parameter Tuning', link: '/usage/parameter-tuning' },
      { text: 'Viewers', link: '/usage/viewers' },
      { text: 'Deployment', link: '/usage/deployment' },
    ],
  },
];

const workflowsSidebar = [
  {
    text: 'Workflows',
    items: [
      { text: 'MuJoCo Warp (MJWP)', link: '/workflows/workflow-mjwp' },
      { text: 'HDMI', link: '/workflows/workflow-hdmi' },
      { text: 'DexMachina', link: '/workflows/workflow-dexmachina' },
    ],
  },
];

const developmentSidebar = [
  {
    text: 'Development',
    items: [
      { text: 'Add a Robot', link: '/development/add-robot' },
      { text: 'Add a Dataset', link: '/development/add-dataset' },
      { text: 'Dataset Release Plan', link: '/development/dataset-release-plan' },
      { text: 'Add a Simulator', link: '/development/add-simulator' },
    ],
  },
];

export default defineConfig({
  lang: 'en-US',
  title: 'SPIDER',
  description: 'Scalable Physics-Informed Dexterous Retargeting',
  base: '/spider/',
  head: [['link', { rel: 'icon', href: '/favicon.ico' }]],
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/quick-start' },
      { text: 'Usage', link: '/usage/data-structure' },
      { text: 'Workflows', link: '/workflows/workflow-mjwp' },
      { text: 'Development', link: '/development/add-robot' },
      { text: 'GitHub', link: 'https://github.com/facebookresearch/spider' },
    ],
    sidebar: {
      '/guide/': guideSidebar,
      '/usage/': usageSidebar,
      '/workflows/': workflowsSidebar,
      '/development/': developmentSidebar,
    },
    socialLinks: [{ icon: 'github', link: 'https://github.com/facebookresearch/spider' }],
  },
});
