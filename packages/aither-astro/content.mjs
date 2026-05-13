import { z } from 'astro/zod';

export const isoDateTimeWithSeconds = z
  .string()
  .regex(
    /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/,
    'Use ISO 8601 with seconds and timezone, e.g. 2026-03-19T16:27:43+08:00',
  );

export const isoDateOnly = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/, 'Use YYYY-MM-DD or full ISO 8601 with timezone');

export const aitherContentDate = z
  .union([isoDateTimeWithSeconds, isoDateOnly])
  .pipe(z.coerce.date());

export function createAitherContentSchema({ image }) {
  return z.object({
    title: z.string(),
    description: z.string().optional(),
    date: aitherContentDate,
    category: z.string().default('General'),
    image: image().optional(),
    tags: z.array(z.string()).optional(),
    sourceTitle: z.string().optional(),
    sourceUrl: z
      .string()
      .refine(
        (value) =>
          value.startsWith('/') || z.url().safeParse(value).success,
        'sourceUrl must be an absolute URL or a site-relative path',
      )
      .optional(),
    sourcePublication: z.string().optional(),
    pinned: z.boolean().default(false),
  });
}
