import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import { satteri, type SatteriProcessorOptions } from '@astrojs/markdown-satteri';
import tailwindcss from '@tailwindcss/vite';
import katex from 'katex';

type HastPluginItem = NonNullable<SatteriProcessorOptions['hastPlugins']>[number];

const satteriKatexPlugin: HastPluginItem = {
  name: 'satteri-katex',
  element: [
    {
      filter: ['pre'],
      visit(node, ctx) {
        const codeChild = node.children?.find(
          (c) => c.type === 'element' && c.tagName === 'code'
        );
        if (!codeChild || codeChild.type !== 'element') return;
        const classes = Array.isArray(codeChild.properties?.className)
          ? codeChild.properties.className
          : [];
        if (!classes.includes('math-display')) return;
        const tex = ctx.textContent(codeChild).trim();
        const html = katex.renderToString(tex, {
          displayMode: true,
          output: 'html',
          throwOnError: false,
        });
        return {
          type: 'element',
          tagName: 'div',
          properties: { 'set:html': html },
          children: [],
        };
      },
    },
    {
      filter: ['code'],
      visit(node, ctx) {
        const classes = Array.isArray(node.properties?.className)
          ? node.properties.className
          : [];
        if (!classes.includes('math-inline')) return;
        const tex = ctx.textContent(node).trim();
        const html = katex.renderToString(tex, {
          displayMode: false,
          output: 'html',
          throwOnError: false,
        });
        return {
          type: 'element',
          tagName: 'span',
          properties: { 'set:html': html },
          children: [],
        };
      },
    },
    {
      filter: ['a'],
      visit(node) {
        const href = node.properties?.href;
        if (typeof href === 'string' && href.startsWith('https://')) {
          return {
            type: 'element',
            tagName: 'a',
            properties: {
              ...node.properties,
              target: '_blank',
              rel: ['noopener', 'noreferrer'],
            },
            children: [...node.children],
          };
        }
      },
    },
  ],
};

const nodeEnv = (
  globalThis as { process?: { env?: Record<string, string | undefined> } }
).process?.env;
const siteUrl = nodeEnv?.ASTRO_SITE || 'https://constcorrectness.github.io';
const basePath = nodeEnv?.ASTRO_BASE || '/';

export default defineConfig({
  site: siteUrl,
  base: basePath,
  output: 'static',
  markdown: {
    syntaxHighlight: {
      type: 'shiki',
      excludeLangs: ['math'],
    },
    processor: satteri({
      features: { math: true },
      hastPlugins: [satteriKatexPlugin],
    }),
  },
  integrations: [react(), mdx()],
  vite: {
    plugins: [tailwindcss()],
  },
});
