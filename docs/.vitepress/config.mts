import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "SPIDER",
  description: "Scalable Physics-Informed DExterous Retargeting",
  base: '/',

  head: [
    ['link', { rel: 'icon', href: '/spider/favicon.ico' }]
  ],

  themeConfig: {
    logo: '/figs/logo.png',

    nav: [
      { text: 'Home', link: '/' },
      {
        text: 'Links',
        items: [
          { text: 'GitHub', link: 'https://github.com/jc-bao/spider' },
          { text: 'Project Page', link: 'https://jc-bao.github.io/spider-project/' }
        ]
      }
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Quick Start', link: '/guide/quick-start' },
          { text: 'Installation', link: '/guide/installation' },
        ]
      },
      {
        text: 'Usage',
        items: [
          { text: 'Viewers', link: '/usage/viewers' },
          { text: 'Data Structure', link: '/usage/data-structure' },
          { text: 'Parameter Tuning', link: '/usage/parameter-tuning' },
          { text: 'Deployment', link: '/usage/deployment' }
        ]
      },
      {
        text: 'Workflows',
        items: [
          { text: 'MJWP Workflow', link: '/workflows/workflow-mjwp' },
          { text: 'DexMachina Workflow', link: '/workflows/workflow-dexmachina' },
          { text: 'HDMI Workflow', link: '/workflows/workflow-hdmi' }
        ]
      },
      {
        text: 'Development',
        items: [
          { text: 'Add New Robot', link: '/development/add-robot' },
          { text: 'Add New Dataset', link: '/development/add-dataset' },
          { text: 'Port New Simulator', link: '/development/add-simulator' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/jc-bao/spider' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2025 SPIDER Contributors'
    },

    search: {
      provider: 'local'
    },

    editLink: {
      pattern: 'https://github.com/jc-bao/spider/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    }
  },

  markdown: {
    lineNumbers: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  ignoreDeadLinks: true
})
