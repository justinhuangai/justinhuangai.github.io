import type { SchemaContext } from 'astro:content';
import type {
  ZodArray,
  ZodBoolean,
  ZodDefault,
  ZodEffects,
  ZodObject,
  ZodOptional,
  ZodString,
  ZodUnion,
} from 'astro/zod';

export declare const isoDateTimeWithSeconds: ZodString;
export declare const isoDateOnly: ZodString;
export declare const aitherContentDate: ZodEffects<
  ZodUnion<[ZodString, ZodString]>,
  Date,
  string
>;

export type AitherContentSchema<TImage extends SchemaContext['image']> = ZodObject<{
  title: ZodString;
  description: ZodOptional<ZodString>;
  date: typeof aitherContentDate;
  category: ZodDefault<ZodString>;
  image: ZodOptional<ReturnType<TImage>>;
  tags: ZodOptional<ZodArray<ZodString>>;
  sourceTitle: ZodOptional<ZodString>;
  sourceUrl: ZodOptional<ZodString>;
  sourcePublication: ZodOptional<ZodString>;
  pinned: ZodDefault<ZodBoolean>;
}>;

export declare function createAitherContentSchema<TImage extends SchemaContext['image']>(
  context: {
    image: TImage;
  },
): AitherContentSchema<TImage>;
