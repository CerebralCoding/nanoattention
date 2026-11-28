// Native Apple ImageIO / CoreGraphics image decoder for nano vision models.
// Zero third-party dependencies; supports PNG, JPEG, WebP, HEIC, TIFF, GIF, BMP
// from local files, in-memory byte buffers, and http(s):// URLs.
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const size_t FM_MAX_REMOTE_IMAGE_BYTES = 32u * 1024u * 1024u;

static int orientation_swaps_axes(int orient) {
    return orient >= 5 && orient <= 8;
}

static CGAffineTransform orientation_transform(int orient, size_t w, size_t h) {
    CGAffineTransform t = CGAffineTransformIdentity;
    switch (orient) {
        case 2: // Flip horizontal
            t = CGAffineTransformTranslate(t, (CGFloat)w, 0);
            t = CGAffineTransformScale(t, -1, 1);
            break;
        case 3: // Rotate 180
            t = CGAffineTransformTranslate(t, (CGFloat)w, (CGFloat)h);
            t = CGAffineTransformRotate(t, (CGFloat)M_PI);
            break;
        case 4: // Flip vertical
            t = CGAffineTransformTranslate(t, 0, (CGFloat)h);
            t = CGAffineTransformScale(t, 1, -1);
            break;
        case 5: // Transpose
            t = CGAffineTransformTranslate(t, 0, (CGFloat)h);
            t = CGAffineTransformRotate(t, (CGFloat)(-M_PI_2));
            t = CGAffineTransformScale(t, -1, 1);
            break;
        case 6: // Rotate 90 CW (in CG bottom-left coords: translate then rotate -pi/2)
            t = CGAffineTransformTranslate(t, 0, (CGFloat)h);
            t = CGAffineTransformRotate(t, (CGFloat)(-M_PI_2));
            break;
        case 7: // Transverse
            t = CGAffineTransformTranslate(t, (CGFloat)w, 0);
            t = CGAffineTransformRotate(t, (CGFloat)M_PI_2);
            t = CGAffineTransformScale(t, -1, 1);
            break;
        case 8: // Rotate 270 CW (90 CCW)
            t = CGAffineTransformTranslate(t, (CGFloat)w, 0);
            t = CGAffineTransformRotate(t, (CGFloat)M_PI_2);
            break;
        default:
            break;
    }
    return t;
}

static int decode_cg_image_source(CGImageSourceRef src, uint8_t **out_rgb, uint32_t *out_w, uint32_t *out_h) {
    if (!src) return 0;

    int orient = 1;
    CFDictionaryRef props = CGImageSourceCopyPropertiesAtIndex(src, 0, NULL);
    if (props) {
        CFNumberRef num = (CFNumberRef)CFDictionaryGetValue(props, kCGImagePropertyOrientation);
        if (num) CFNumberGetValue(num, kCFNumberIntType, &orient);
        CFRelease(props);
    }

    CGImageRef image = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    if (!image) return 0;

    size_t raw_w = CGImageGetWidth(image);
    size_t raw_h = CGImageGetHeight(image);
    if (raw_w == 0 || raw_h == 0 || raw_w > 32768 || raw_h > 32768) {
        CGImageRelease(image);
        return 0;
    }

    size_t w = orientation_swaps_axes(orient) ? raw_h : raw_w;
    size_t h = orientation_swaps_axes(orient) ? raw_w : raw_h;

    size_t rgba_bytes = w * h * 4;
    uint8_t *rgba = (uint8_t *)calloc(rgba_bytes, 1);
    if (!rgba) {
        CGImageRelease(image);
        return 0;
    }

    CGImageAlphaInfo alpha = CGImageGetAlphaInfo(image);
    int has_alpha = (alpha == kCGImageAlphaPremultipliedLast ||
                     alpha == kCGImageAlphaPremultipliedFirst ||
                     alpha == kCGImageAlphaLast ||
                     alpha == kCGImageAlphaFirst ||
                     alpha == kCGImageAlphaOnly);
    if (has_alpha) {
        for (size_t y = 0; y < h; y++) {
            for (size_t x = 0; x < w; x++) {
                uint8_t bg = (((x / 8) + (y / 8)) % 2 == 0) ? 204 : 153;
                size_t off = (y * w + x) * 4;
                rgba[off + 0] = bg;
                rgba[off + 1] = bg;
                rgba[off + 2] = bg;
                rgba[off + 3] = 255;
            }
        }
    }

    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (!cs) cs = CGColorSpaceCreateDeviceRGB();
    CGBitmapInfo bi = (CGBitmapInfo)(kCGBitmapByteOrder32Big | kCGImageAlphaNoneSkipLast);
    CGContextRef ctx = CGBitmapContextCreate(rgba, w, h, 8, w * 4, cs, bi);
    CGColorSpaceRelease(cs);
    if (!ctx) {
        free(rgba);
        CGImageRelease(image);
        return 0;
    }

    CGContextSetInterpolationQuality(ctx, kCGInterpolationNone);
    if (orient != 1) {
        CGAffineTransform t = orientation_transform(orient, w, h);
        CGContextConcatCTM(ctx, t);
    }
    CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)raw_w, (CGFloat)raw_h), image);
    CGContextRelease(ctx);
    CGImageRelease(image);

    uint8_t *rgb = (uint8_t *)malloc(w * h * 3);
    if (!rgb) {
        free(rgba);
        return 0;
    }
    size_t n_px = w * h;
    for (size_t i = 0; i < n_px; i++) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    free(rgba);

    *out_rgb = rgb;
    *out_w = (uint32_t)w;
    *out_h = (uint32_t)h;
    return 1;
}

int fm_load_image_rgb(const char *path, uint8_t **out_rgb, uint32_t *out_w, uint32_t *out_h) {
    if (!path || !out_rgb || !out_w || !out_h) return 0;
    *out_rgb = NULL;
    *out_w = 0;
    *out_h = 0;

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSString *nsPath = [NSString stringWithUTF8String:path];
    if (!nsPath) {
        [pool drain];
        return 0;
    }
    NSURL *url = [NSURL fileURLWithPath:nsPath];
    CGImageSourceRef src = CGImageSourceCreateWithURL((CFURLRef)url, NULL);
    int ok = decode_cg_image_source(src, out_rgb, out_w, out_h);
    if (src) CFRelease(src);
    [pool drain];
    return ok;
}

int fm_load_image_rgb_bytes(const uint8_t *data, size_t len, uint8_t **out_rgb, uint32_t *out_w, uint32_t *out_h) {
    if (!data || len == 0 || !out_rgb || !out_w || !out_h) return 0;
    *out_rgb = NULL;
    *out_w = 0;
    *out_h = 0;

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    CFDataRef cfData = CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, data, (CFIndex)len, kCFAllocatorNull);
    if (!cfData) {
        [pool drain];
        return 0;
    }
    CGImageSourceRef src = CGImageSourceCreateWithData(cfData, NULL);
    int ok = decode_cg_image_source(src, out_rgb, out_w, out_h);
    if (src) CFRelease(src);
    CFRelease(cfData);
    [pool drain];
    return ok;
}

int fm_load_image_rgb_url(const char *url_str, uint8_t **out_rgb, uint32_t *out_w, uint32_t *out_h) {
    if (!url_str || !out_rgb || !out_w || !out_h) return 0;
    *out_rgb = NULL;
    *out_w = 0;
    *out_h = 0;

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSString *nsUrl = [NSString stringWithUTF8String:url_str];
    if (!nsUrl) {
        [pool drain];
        return 0;
    }
    NSURL *url = [NSURL URLWithString:nsUrl];
    NSString *scheme = [[url scheme] lowercaseString];
    if (!url || (![scheme isEqualToString:@"http"] && ![scheme isEqualToString:@"https"])) {
        [pool drain];
        return 0;
    }

    NSURLSessionConfiguration *cfg = [NSURLSessionConfiguration ephemeralSessionConfiguration];
    cfg.timeoutIntervalForRequest = 30.0;
    cfg.timeoutIntervalForResource = 60.0;
    cfg.URLCache = nil;
    cfg.HTTPCookieStorage = nil;
    NSURLSession *session = [NSURLSession sessionWithConfiguration:cfg];

    __block NSData *fetched = nil;
    __block int fetch_ok = 0;
    NSConditionLock *lock = [[NSConditionLock alloc] initWithCondition:0];

    NSURLSessionDataTask *task = [session dataTaskWithURL:url completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (!error && data && [data length] > 0 && [data length] <= FM_MAX_REMOTE_IMAGE_BYTES) {
            if ([response isKindOfClass:[NSHTTPURLResponse class]]) {
                NSInteger code = [(NSHTTPURLResponse *)response statusCode];
                if (code >= 200 && code < 300) {
                    fetched = [data retain];
                    fetch_ok = 1;
                }
            }
        }
        [lock lock];
        [lock unlockWithCondition:1];
    }];
    [task resume];

    BOOL finished = [lock lockWhenCondition:1 beforeDate:[NSDate dateWithTimeIntervalSinceNow:65.0]];
    if (finished) {
        [lock unlock];
    } else {
        [task cancel];
    }
    [session invalidateAndCancel];
    [lock release];

    int ok = 0;
    if (finished && fetch_ok && fetched) {
        ok = fm_load_image_rgb_bytes((const uint8_t *)[fetched bytes], [fetched length], out_rgb, out_w, out_h);
    }
    if (fetched) [fetched release];
    [pool drain];
    return ok;
}

void fm_free_image(uint8_t *rgb) {
    free(rgb);
}
